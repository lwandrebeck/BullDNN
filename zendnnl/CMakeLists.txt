# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

cmake_minimum_required(VERSION 3.26)

# project information
project(ZENDNNL_LIBRARY VERSION ${ZENDNNL_PROJECT_VERSION})

# library version
set(ZENDNNL_LIBRARY_VERSION ${PROJECT_VERSION})
set(ZENDNNL_LIBRARY_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(ZENDNNL_LIBRARY_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(ZENDNNL_LIBRARY_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# library components
set(ZENDNNL_LIB_BUILD_ARCHIVE ON CACHE BOOL "build zendnnl archive")
set(ZENDNNL_LIB_BUILD_SHARED OFF CACHE BOOL "build zendnnl shared lib")

# build integration mode
set(ZENDNNL_FWK_BUILD OFF CACHE BOOL "framework build mode")

# library dependencies
set(ZENDNNL_AOCLDLP_INJECTED OFF CACHE BOOL "aocldlp injected")
set(ZENDNNL_ONEDNN_INJECTED  OFF CACHE BOOL "onednn injected")
set(ZENDNNL_LIBXSMM_INJECTED  OFF CACHE BOOL "libxsmm injected")
set(ZENDNNL_PARLOOPER_INJECTED  OFF CACHE BOOL "parlooper injected")
set(ZENDNNL_FBGEMM_INJECTED  OFF CACHE BOOL "fbgemm injected")

# cmake variables
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "verbose makefile")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_MESSAGE_LOG_LEVEL "DEBUG")
set(CMAKE_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# informative messages
message(STATUS
  "${ZENDNNL_MSG_PREFIX} ZENDNNL_LIBRARY_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "ZENDNNL_LIBRARY_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  message(STATUS "building zendnnl library archive")
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  message(STATUS "building zendnnl library shared lib")
endif()

# include zendnnl files
include(CheckCXXSourceCompiles)
include(ZenDnnlMacros)
include(ZenDnnlDependenciesInfo)

# global cxx flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
# Target CPU architecture selection.
#   zen (default) : upstream ZenDNN behavior (AVX2 + AVX-512), full compatibility.
#   bdver1        : AMD family 15h Bulldozer   (AVX, FMA4, XOP; no FMA3/AVX2/AVX-512).
#   bdver2        : AMD family 15h Piledriver  (adds FMA3, F16C, BMI1; no AVX2).
#   bdver3        : AMD family 15h Steamroller (as Piledriver; no AVX2).
#   bdver4        : AMD family 15h Excavator   (adds AVX2, BMI2; no AVX-512).
#   native        : tune for the build host.
# AVX-512 exists nowhere in family 15h; AVX2 only on Excavator (bdver4).
set(BULLDNN_TARGET_ARCH "zen" CACHE STRING
  "Target CPU arch: zen | bdver1 | bdver2 | bdver3 | bdver4 | native")
set_property(CACHE BULLDNN_TARGET_ARCH PROPERTY STRINGS
  zen bdver1 bdver2 bdver3 bdver4 native)

if(BULLDNN_TARGET_ARCH STREQUAL "zen")
  # Upstream default: AVX-512-capable Zen cores.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mavx512f -mavx512vl -mavx512bf16")
  endif()
  message(STATUS "${ZENDNNL_MSG_PREFIX} BullDNN target arch: zen (upstream, AVX-512)")
elseif(BULLDNN_TARGET_ARCH STREQUAL "native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  message(STATUS "${ZENDNNL_MSG_PREFIX} BullDNN target arch: native (host-tuned)")
else()
  # AMD family 15h. -march=bdverN pins the baseline to that core's exact ISA,
  # so the compiler can never emit an unsupported instruction. AVX-512 kernels
  # guarded by __attribute__((target("avx512..."))) still compile but are never
  # dispatched at runtime (gated by platform_info feature detection).
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${BULLDNN_TARGET_ARCH}")
  message(STATUS "${ZENDNNL_MSG_PREFIX} BullDNN target arch: ${BULLDNN_TARGET_ARCH} (AMD family 15h, no AVX-512)")
endif()

check_cxx_source_compiles("
  #include <immintrin.h>
  __attribute__((target(\"avx512f,avx512vl,avx512bw,avx512fp16\")))
  void test(const void *src, void *dst, __mmask32 k, __m512h v) {
    (void)_mm512_maskz_loadu_ph(k, src);
    _mm512_mask_storeu_ph(dst, k, v);
  }
  int main() { return 0; }
" ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)

if(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)
  add_compile_definitions(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)
endif()

# find common packages
zendnnl_required_packages()

# find required dependencies
find_build_dependencies(${CMAKE_INSTALL_PREFIX})

# build library
add_subdirectory(src)

# build gtests
add_subdirectory(gtests)

# create version file
set(ZENDNNL_VERSION_FILE "${CMAKE_BINARY_DIR}/zendnnl_version.hpp")
create_version_header_file(${ZENDNNL_VERSION_FILE}
  ${ZENDNNL_LIBRARY_VERSION_MAJOR}
  ${ZENDNNL_LIBRARY_VERSION_MINOR}
  ${ZENDNNL_LIBRARY_VERSION_PATCH})

# install config file
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/config/zendnnl_user_config.json"
  DESTINATION zendnnl/config)
# install version file
install(FILES "${ZENDNNL_VERSION_FILE}"
  DESTINATION zendnnl/include)
