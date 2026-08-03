#!/usr/bin/env bash
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
# Usage:
#   scripts/check_format.sh --check                  # Check all files
#   scripts/check_format.sh --fix                    # Auto-format all files
#   scripts/check_format.sh --check file1.cpp file2.h  # Check specific files
#   scripts/check_format.sh --fix src/foo.cpp          # Auto-format specific files
#   scripts/check_format.sh --fix src/dir              # Auto-format tracked source files under a directory

set -euo pipefail

CLANG_FORMAT="${CLANG_FORMAT:-clang-format-18}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

if ! command -v "$CLANG_FORMAT" &>/dev/null; then
    echo "ERROR: $CLANG_FORMAT not found. Install clang-format >= 18.1.8:"
    echo "  pip install clang-format==18.1.8"
    echo "  OR set CLANG_FORMAT env var to point to the binary."
    exit 1
fi

REQUIRED_VERSION="18.1.8"
CF_VERSION=$("$CLANG_FORMAT" --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)

version_lt() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" != "$2" ]
}

if version_lt "$CF_VERSION" "$REQUIRED_VERSION"; then
    echo "ERROR: clang-format version $CF_VERSION is too old (need >= $REQUIRED_VERSION)."
    echo "  pip install clang-format==18.1.8"
    exit 1
fi

MODE="${1:---check}"
shift || true

add_file() {
    FILES="${FILES:+$FILES
}$1"
}

add_directory() {
    local dir="$1"
    local abs_dir rel_dir

    abs_dir="$(realpath "$dir")"
    rel_dir="${abs_dir#$REPO_ROOT/}"

    git -C "$REPO_ROOT" ls-files -- "$rel_dir" | while IFS= read -r f; do
        case "$f" in
            *.c|*.cpp|*.cc|*.h|*.hpp|*.hxx|*.cu)
                [ -f "$REPO_ROOT/$f" ] && echo "$REPO_ROOT/$f"
                ;;
        esac
    done
}

# If file or directory paths are provided as arguments, use those; otherwise auto-discover.
if [ $# -gt 0 ]; then
    FILES=""
    for arg in "$@"; do
        if [ -f "$arg" ]; then
            add_file "$(realpath "$arg")"
        elif [ -f "$REPO_ROOT/$arg" ]; then
            add_file "$(realpath "$REPO_ROOT/$arg")"
        elif [ -d "$arg" ]; then
            DIR_FILES="$(add_directory "$arg")"
            FILES="${FILES:+$FILES
}$DIR_FILES"
        elif [ -d "$REPO_ROOT/$arg" ]; then
            DIR_FILES="$(add_directory "$REPO_ROOT/$arg")"
            FILES="${FILES:+$FILES
}$DIR_FILES"
        else
            echo "WARNING: File not found: $arg (skipping)"
        fi
    done
elif [ -n "${BASE_SHA:-}" ] && [ -n "${HEAD_SHA:-}" ]; then
    FILES=$(git diff --name-only --diff-filter=ACMR "$BASE_SHA" "$HEAD_SHA" -- \
        '*.c' '*.cpp' '*.cc' '*.h' '*.hpp' '*.hxx' '*.cu' | \
        while IFS= read -r f; do [ -f "$REPO_ROOT/$f" ] && echo "$REPO_ROOT/$f"; done || true)
else
    FILES=$(git ls-files -- '*.c' '*.cpp' '*.cc' '*.h' '*.hpp' '*.hxx' '*.cu' | \
        while IFS= read -r f; do [ -f "$REPO_ROOT/$f" ] && echo "$REPO_ROOT/$f"; done || true)
fi

# Filter out paths matching .clang-format-ignore
IGNORE_FILE="$REPO_ROOT/.clang-format-ignore"
if [ -f "$IGNORE_FILE" ] && [ -n "$FILES" ]; then
    FILTERED=""
    while IFS= read -r pattern; do
        [[ "$pattern" =~ ^#.*$ || -z "$pattern" ]] && continue
        EXCLUDE_PATTERNS+=("$pattern")
    done < "$IGNORE_FILE"

    while IFS= read -r file; do
        REL_PATH="${file#$REPO_ROOT/}"
        SKIP=false
        for pattern in "${EXCLUDE_PATTERNS[@]:-}"; do
            case "$REL_PATH" in
                $pattern*) SKIP=true; break ;;
            esac
            # Handle **/ prefix patterns
            case "$REL_PATH" in
                */${pattern#\*\*/}*) SKIP=true; break ;;
            esac
        done
        if [ "$SKIP" = false ]; then
            FILTERED="${FILTERED:+$FILTERED
}$file"
        fi
    done <<< "$FILES"
    FILES="$FILTERED"
fi

if [ -z "$FILES" ]; then
    echo "No source files to check."
    exit 0
fi

case "$MODE" in
    --check)
        echo "Checking formatting (dry-run)..."
        FAILED=0
        while IFS= read -r file; do
            if ! "$CLANG_FORMAT" --dry-run --Werror "$file" 2>/dev/null; then
                echo "  FAIL: $file"
                FAILED=1
            fi
        done <<< "$FILES"

        if [ "$FAILED" -eq 1 ]; then
            echo ""
            echo "Formatting check FAILED. Run 'scripts/check_format.sh --fix' to auto-format."
            exit 1
        fi
        echo "All files are correctly formatted."
        ;;
    --fix)
        echo "Formatting files in-place..."
        while IFS= read -r file; do
            "$CLANG_FORMAT" -i "$file"
            echo "  formatted: $file"
        done <<< "$FILES"
        echo "Done."
        ;;
    *)
        echo "Usage: $0 [--check|--fix]"
        exit 1
        ;;
esac
