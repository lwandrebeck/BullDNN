/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/auto_tuner/auto_tuner.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

/** Environment variable for comma-separated autotuner algo candidates (int values). */
const char *ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES
        = "ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES";

static std::string trim_whitespace(const std::string &s) {
    auto start = s.find_first_not_of(" \t");
    if (start == std::string::npos) { return ""; }
    auto end = s.find_last_not_of(" \t");
    return s.substr(start, end - start + 1);
}

/**
 * Parses ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES env var.
 * Format: comma-separated integers matching matmul_algo_t enum values
 * (e.g. "1,2,6" for aocl_dlp_blocked, onednn_blocked, libxsmm).
 * Valid range: 0 to algo_count-1. dynamic_dispatch (0) and auto_tuner (8) are skipped.
 * Returns empty vector if unset or if no valid algos were parsed.
 */
static std::vector<matmul_algo_t> parse_algo_candidates_from_env() {
    const char *val = std::getenv(ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES);
    if (!val || val[0] == '\0') return {};

    const int32_t algo_count_val
            = static_cast<int32_t>(matmul_algo_t::algo_count);
    std::vector<matmul_algo_t> result;
    // Track ids already accepted so duplicates in the env string don't inflate
    // num_algo or bias the round-robin in the eval loop.
    std::vector<bool> seen(static_cast<size_t>(algo_count_val), false);
    std::string str(val);
    std::istringstream iss(str);
    std::string token;

    while (std::getline(iss, token, ',')) {
        std::string trimmed = trim_whitespace(token);
        if (trimmed.empty()) { continue; }

        try {
            size_t pos = 0;
            int32_t ival = std::stoi(trimmed, &pos);
            if (pos != trimmed.size()) {
                log_info("AutoTuner: invalid algo value '", trimmed,
                        "' (trailing characters) in ",
                        ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES, ", skipping");
                continue;
            }
            if (ival < 0 || ival >= algo_count_val) {
                log_info("AutoTuner: algo id ", ival, " out of range [0,",
                        algo_count_val - 1, "] in ",
                        ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES, ", skipping");
                continue;
            }
            matmul_algo_t algo = static_cast<matmul_algo_t>(ival);
            if (algo == matmul_algo_t::auto_tuner
                    || algo == matmul_algo_t::dynamic_dispatch) {
                log_info("AutoTuner: algo id ", ival,
                        " invalid as candidate, skipping");
                continue;
            }
            if (seen[static_cast<size_t>(ival)]) {
                log_info("AutoTuner: duplicate algo id ", ival, " in ",
                        ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES, ", skipping");
                continue;
            }
            seen[static_cast<size_t>(ival)] = true;
            result.push_back(algo);
        } catch (const std::exception &) {
            log_info("AutoTuner: invalid algo value '", trimmed, "' in ",
                    ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES, ", skipping");
        }
    }
    return result;
}

} // anonymous namespace

// Returns the active candidate list, parsing ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES
// exactly once on the first call (parse errors are logged that one time) and
// falling back to a built-in default when the env var is unset or empty.
// Magic-statics initialization is thread-safe in C++11+, so the returned
// const reference is safe to share across threads.
//
// The default list must only contain backends this build can actually execute.
// A candidate that returns without computing is not merely useless: the
// evaluate phase below keeps whichever candidate timed fastest, and "did not
// compute" times as near-zero, so an unavailable backend would win the
// comparison and be cached as the best algorithm for that shape.
const std::vector<matmul_algo_t> &get_algo_candidates() {
    static const std::vector<matmul_algo_t> candidates = []() {
        auto env_candidates = parse_algo_candidates_from_env();
        if (!env_candidates.empty()) { return env_candidates; }
#if ZENDNNL_DEPENDS_AOCLDLP
        return std::vector<matmul_algo_t> {
                matmul_algo_t::aocl_dlp_blocked,
                matmul_algo_t::onednn_blocked,
        };
#else
        // No AOCL-DLP in this build, so aocl_dlp_blocked can never compute and
        // must not be offered. Offer every backend this build does contain: no
        // single one covers all dtypes on a host without AVX-512, so the list
        // has to span them. On an Excavator part, for example, oneDNN serves f32
        // but cannot create a BF16 primitive, while libxsmm serves BF16 but
        // declines f32.
        //
        // Candidates that decline a given problem are safe here because
        // run_first_computing_candidate() moves on to the next one and the
        // declined attempt is neither timed nor cached.
        std::vector<matmul_algo_t> selected;
#if ZENDNNL_DEPENDS_ONEDNN
        selected.push_back(matmul_algo_t::onednn_blocked);
#endif
#if ZENDNNL_DEPENDS_LIBXSMM
        // matmul_algo_t::libxsmm, not libxsmm_blocked: the libxsmm branch in
        // matmul_kernel_wrapper() tests for the former only, so the blocked
        // spelling reaches the partitioner instead and declines the shapes we
        // need it for. Measured on an A10-8770E, BF16 512x512x512: algo
        // libxsmm computes at ~39 GFLOPS, libxsmm_blocked declines.
        selected.push_back(matmul_algo_t::libxsmm);
#endif
        // Always last, and always present: the native kernels need no external
        // dependency, so they keep the list non-empty even in a build with
        // neither oneDNN nor libxsmm.
        selected.push_back(matmul_algo_t::native_gemm);
        return selected;
#endif
    }();
    return candidates;
}

inline matmul_algo_t get_algo(
        int index, const std::vector<matmul_algo_t> &candidates) {
    return candidates[index % candidates.size()];
}

// True when a dispatched call came back marked as the AOCL-DLP fallback in a
// build that has no AOCL-DLP. matmul_kernel_wrapper() takes `kernel` by
// reference and rewrites it to aocl_dlp when a backend declined without
// computing, so this is the dispatch's existing "did not compute" signal.
//
// The tuner must never time or cache such a call: doing nothing is the fastest
// thing a candidate can do, so an algorithm that cannot run on this host would
// win every comparison and be cached as the best one for that shape. Where
// AOCL-DLP is present the fallback genuinely computed, so this is always false
// and every decision below behaves exactly as before.
inline bool did_not_compute(matmul_algo_t executed) {
#if ZENDNNL_DEPENDS_AOCLDLP
    (void)executed;
    return false;
#else
    return executed == matmul_algo_t::aocl_dlp
            || executed == matmul_algo_t::aocl_dlp_blocked
            || executed == matmul_algo_t::batched_sgemm;
#endif
}

// Hash of the dtype combination, folded into the tuner's key.
//
// Key_matmul carries shape, leading dimensions, transposes and the weight
// pointer, but no dtypes -- so an f32 and a BF16 matmul of the same shape shared
// one entry, and therefore one "best algorithm". On a host without AVX-512 those
// two do not even share a backend: oneDNN serves f32 and cannot create a BF16
// primitive, while libxsmm serves BF16 and declines f32. A single shared entry
// therefore recorded whichever dtype ran last and made the tuner fight itself,
// re-selecting on every dtype change instead of settling.
//
// extra_input_hash exists for exactly this kind of extra discriminator and is
// already part of Key_matmul's operator== and its std::hash specialization, so
// no shared key structure has to change.
inline size_t hash_dtypes(const matmul_data_types &dtypes) {
    size_t seed = 0;
    seed = zendnnl::common::hash_combine(
            seed, static_cast<uint64_t>(dtypes.src));
    seed = zendnnl::common::hash_combine(
            seed, static_cast<uint64_t>(dtypes.wei));
    seed = zendnnl::common::hash_combine(
            seed, static_cast<uint64_t>(dtypes.dst));
    seed = zendnnl::common::hash_combine(
            seed, static_cast<uint64_t>(dtypes.bias));
    seed = zendnnl::common::hash_combine(
            seed, static_cast<uint64_t>(dtypes.compute));
    return seed;
}

// Runs `first`, then each remaining candidate in turn until one actually
// computes, and returns the algorithm that did. `run` executes a single
// candidate and returns what the dispatch reports having executed.
//
// This is what lets the candidate list hold algorithms that only cover some
// dtypes. Without it, a candidate that cannot serve the current problem would
// leave the output untouched and the caller would report failure -- which the
// skip phase would do on its very first call, before any timing exists.
// Retrying costs nothing because a declining backend computes nothing.
// `computed` is set false only when no candidate could run the problem.
template <typename RunFn>
matmul_algo_t run_first_computing_candidate(matmul_algo_t first,
        const std::vector<matmul_algo_t> &candidates, RunFn &&run,
        bool &computed) {
    matmul_algo_t executed = run(first);
    if (!did_not_compute(executed)) {
        computed = true;
        return executed;
    }
    for (matmul_algo_t candidate : candidates) {
        if (candidate == first) { continue; }
        apilog_info("AutoTuner: algo ", static_cast<int32_t>(first),
                " could not run this problem; trying algo ",
                static_cast<int32_t>(candidate));
        executed = run(candidate);
        if (!did_not_compute(executed)) {
            computed = true;
            return executed;
        }
    }
    computed = false;
    return executed;
}

/**
 * @brief Maps an M dimension value to its bin representative.
 *
 * Bins M into fixed-size ranges so that nearby M values share the same
 * autotuner key. For bin_size B, M is mapped to ceil(M / B) * B.
 * A bin_size of 1 disables binning (identity mapping).
 *
 * Bin size is controlled via ZENDNNL_AUTO_BIN_SIZE (default: 16). Invalid
 * values (non-numeric, negative, zero, or out of range) are ignored and the
 * default is used. Parsing uses strtoul rather than std::stoi so a malformed
 * env var cannot throw during static initialization and terminate the
 * process.
 */
unsigned int get_binned_m(unsigned int M) {
    static unsigned int bin_size = []() -> unsigned int {
        constexpr unsigned int kDefault = 16;
        const char *env = std::getenv("ZENDNNL_AUTO_BIN_SIZE");
        if (!env || env[0] == '\0') { return kDefault; }
        errno = 0;
        char *end = nullptr;
        unsigned long parsed = std::strtoul(env, &end, 10);
        // Reject: parse error (no digits consumed), trailing garbage, overflow,
        // or zero (which would disable binning silently). Fall back to the
        // documented default rather than throwing.
        if (end == env || *end != '\0' || errno == ERANGE || parsed == 0
                || parsed > UINT_MAX) {
            return kDefault;
        }
        return static_cast<unsigned int>(parsed);
    }();
    if (bin_size == 1) { return M; }
    return ((M + bin_size - 1) / bin_size) * bin_size;
}

unsigned int get_auto_tuner_iter(std::string val, bool is_skip) {
    char *skip_env_var = std::getenv(val.c_str());
    if (skip_env_var) { return std::stoi(skip_env_var); }
    return is_skip ? MATMUL_SKIP_ITER : MATMUL_EVALUATE_ITER;
}

matmul_algo_t auto_compute_matmul_v1(char layout, char transA, char transB,
        int M, int N, int K, float alpha, const void *A, int lda, const void *B,
        int ldb, float beta, void *C, int ldc, matmul_data_types &dtypes,
        zendnnl::ops::matmul_algo_t kernel, char mem_format_a,
        char mem_format_b, matmul_params &lowoha_param,
        matmul_batch_params_t &batch_params, const void *bias,
        bool is_weights_const, size_t src_type_size, size_t out_type_size,
        int num_threads) {

    //Simplified Map having Key as struct and value as Algo.
    static std::unordered_map<Key_matmul, matmul_algo_t> matmul_kernel_map;

    //Map value is tuple of (iteration count, execution time of algo, Algo Path)
    static std::unordered_map<Key_matmul,
            std::tuple<unsigned int, float, matmul_algo_t>>
            matmul_kernel_map1_helper;

    unsigned int binned_m = get_binned_m(M);

    Key_matmul key_obj_auto(transA, transB, binned_m, K, N, lda, ldb, B,
            (int32_t)matmul_algo_t::none, hash_dtypes(dtypes));

    double cur_algo_time = 0.0;
    key_obj_auto.weights = is_weights_const ? B : nullptr;

    profiler_t profiler;

    const auto &candidates = get_algo_candidates();
    const auto num_algo = candidates.size();

    // Executes one candidate and returns the algorithm the dispatch reports
    // having run. matmul_kernel_wrapper() takes `kernel` by reference and
    // rewrites it when a backend declines, so the returned value is what
    // actually executed, not what was asked for.
    auto run_algo = [&](matmul_algo_t candidate) {
        matmul_algo_t executed = candidate;
        // Go through execute_selected_algo(), not matmul_kernel_wrapper(), so a
        // candidate is measured and run exactly as it would be if the caller had
        // requested it by name -- mm partitioner included. Calling the wrapper
        // directly skipped the partitioner, so libxsmm was both timed and
        // executed in its slower unpartitioned form.
        execute_selected_algo(layout, transA == 't', transB == 't', M, N, K,
                alpha, A, lda, B, ldb, bias, beta, C, ldc, is_weights_const,
                src_type_size, out_type_size, num_threads, executed,
                lowoha_param, batch_params);
        return executed;
    };

    static const unsigned int base_skip_iter
            = get_auto_tuner_iter("ZENDNNL_MATMUL_SKIP_ITER", true);
    static const unsigned int base_evaluate_iter
            = get_auto_tuner_iter("ZENDNNL_MATMUL_EVAL_ITER", false);

    // Ensure at least one full round-robin per phase.
    const unsigned int skip_iter
            = std::max(base_skip_iter, static_cast<unsigned int>(num_algo));
    const unsigned int evaluate_iter
            = std::max(base_evaluate_iter, static_cast<unsigned int>(num_algo));

    // Log adjustment at most once per process. auto_compute_matmul_v1 can be
    // entered concurrently (the matmul_kernel_map* accesses below are guarded
    // by get_lowoha_mutex()), so an atomic flag with a single CAS keeps the
    // read/write race-free without taking the coarse mutex just for logging.
    static std::atomic<bool> logged_algo_counts {false};
    if (skip_iter != base_skip_iter || evaluate_iter != base_evaluate_iter) {
        bool expected = false;
        if (logged_algo_counts.compare_exchange_strong(
                    expected, true, std::memory_order_relaxed)) {
            apilog_verbose("AutoTuner: adjusted skip_iter ", base_skip_iter,
                    "->", skip_iter, ", eval_iter ", base_evaluate_iter, "->",
                    evaluate_iter);
        }
    }

    //finds object in map
    auto found_obj = matmul_kernel_map1_helper.find(key_obj_auto);

    //If current iterations less than Skip iteration then run default algo.
    //Checks using the (0) element of map value that denotes count of iterations in map
    if (found_obj == matmul_kernel_map1_helper.end()
            || std::get<0>(found_obj->second) < skip_iter) {

        apilog_info("AutoTuner SKIP Iteration");

        //If Key not found in map then time the algo and add new element to map
        if (found_obj == matmul_kernel_map1_helper.end()) {

            //Time start
            profiler.tbp_start();

            bool computed = false;
            kernel = run_first_computing_candidate(
                    candidates[0], candidates, run_algo, computed);
            //Time end
            profiler.tbp_stop();
            cur_algo_time = profiler.tbp_elapsedtime();

            //Create new entry
            get_lowoha_mutex().lock();
            //Map value is tuple of (iteration count, execution time of algo, Algo Path)
            // Seed the best-time field with the largest float when nothing
            // computed. Seeding it with the ~0 ms of a call that did no work
            // would make this entry unbeatable for the life of the process, so
            // no real measurement could ever replace it.
            matmul_kernel_map1_helper[key_obj_auto]
                    = {1,
                            computed ? cur_algo_time
                                     : std::numeric_limits<float>::max(),
                            kernel};
            //Simplified Map having Key as struct and value as Algo.
            matmul_kernel_map[key_obj_auto] = kernel;
            get_lowoha_mutex().unlock();
        }
        //If key found then increment the iter_count and run next algo.
        else {
            get_lowoha_mutex().lock();
            matmul_algo_t skip_candidate
                    = get_algo(std::get<0>(found_obj->second), candidates);
            std::get<0>(found_obj->second) += 1;
            get_lowoha_mutex().unlock();
            bool computed = false;
            kernel = run_first_computing_candidate(
                    skip_candidate, candidates, run_algo, computed);
        }
    }
    //Read Value from map.
    //Runs after skip iterations and evaluation iterations are done.
    else if (std::get<0>(found_obj->second) == evaluate_iter + skip_iter) {
        //Get best algo for given layer from MAP
        kernel = matmul_kernel_map[key_obj_auto];

        bool computed = false;
        kernel = run_first_computing_candidate(
                kernel, candidates, run_algo, computed);
    }
    //Updates the map values by running different algorithms
    else {

        //Get the number of iteration already ran and select Algo to run for current iteration
        get_lowoha_mutex().lock();
        auto &state = found_obj->second;
        unsigned int iter_count = std::get<0>(state);
        // Evaluate phase restarts round-robin from candidates[0]
        unsigned int eval_index = iter_count - skip_iter;
        matmul_algo_t eval_candidate = get_algo(eval_index, candidates);
        std::get<0>(state) += 1;
        get_lowoha_mutex().unlock();
        //Time start
        profiler.tbp_start();

        bool computed = false;
        kernel = run_first_computing_candidate(
                eval_candidate, candidates, run_algo, computed);

        //Time end
        profiler.tbp_stop();
        cur_algo_time = profiler.tbp_elapsedtime();
        apilog_info("AutoTuner Evaluate Iteration algo:", (int32_t)kernel,
                " time:", cur_algo_time);
        //If current run gives better timing then update
        get_lowoha_mutex().lock();
        // Only a call that computed carries a usable time. The measurement spans
        // any declined attempt preceding the successful one, but a decline
        // returns without doing work, so the distortion is negligible.
        if (computed && cur_algo_time < std::get<1>(state)) {
            std::get<1>(state) = cur_algo_time; //Minimum time for chosen algo
            std::get<2>(state) = kernel;
            matmul_kernel_map[key_obj_auto] = kernel;
        }
        get_lowoha_mutex().unlock();
    }
    return kernel;
}

matmul_algo_t auto_compute_matmul_v2(char layout, char transA, char transB,
        int M, int N, int K, float alpha, const void *A, int lda, const void *B,
        int ldb, float beta, void *C, int ldc, matmul_data_types &dtypes,
        zendnnl::ops::matmul_algo_t kernel, char mem_format_a,
        char mem_format_b, matmul_params &lowoha_param,
        matmul_batch_params_t &batch_params, const void *bias,
        bool is_weights_const, size_t src_type_size, size_t out_type_size,
        int num_threads) {
    // Per-layer iteration count drives this layer's phase (skip/eval/inference)
    // and selects the eval slot for its current call. The map is keyed by the
    // matmul shape; layers with identical shapes share an entry, which is
    // intentional because they would also share the chosen algo.
    static std::unordered_map<Key_matmul, unsigned int> per_layer_iter_count;

    static matmul_algo_t global_best_algo = matmul_algo_t::none;
    static bool global_best_computed = false;

    Key_matmul key_obj_auto(transA, transB, M, K, N, lda, ldb, B,
            (int32_t)matmul_algo_t::none, hash_dtypes(dtypes));

    key_obj_auto.weights = is_weights_const ? B : nullptr;

    const auto &candidates = get_algo_candidates();
    const auto num_algo = candidates.size();

    // Read defaults once; these are never mutated after initialization.
    static const unsigned int base_skip_iter
            = get_auto_tuner_iter("ZENDNNL_MATMUL_SKIP_ITER", true);
    static const unsigned int base_evaluate_iter
            = get_auto_tuner_iter("ZENDNNL_MATMUL_EVAL_ITER", false);

    // Ensure at least one full round-robin per phase.
    // Computed per-call (not static) using the current process-global algo
    // candidate count from get_algo_candidates().
    const unsigned int skip_iter
            = std::max(base_skip_iter, static_cast<unsigned int>(num_algo));
    const unsigned int evaluate_iter
            = std::max(base_evaluate_iter, static_cast<unsigned int>(num_algo));

    // Log adjustment at most once per process. auto_compute_matmul_v2 can be
    // entered concurrently (the static map/vector accesses below are guarded
    // by get_lowoha_mutex()), so an atomic flag with a single CAS keeps the
    // read/write race-free without taking the coarse mutex just for logging.
    static std::atomic<bool> logged_algo_counts {false};
    if (skip_iter != base_skip_iter || evaluate_iter != base_evaluate_iter) {
        bool expected = false;
        if (logged_algo_counts.compare_exchange_strong(
                    expected, true, std::memory_order_relaxed)) {
            apilog_verbose("AutoTuner: adjusted skip_iter ", base_skip_iter,
                    "->", skip_iter, ", eval_iter ", base_evaluate_iter, "->",
                    evaluate_iter);
        }
    }

    static std::vector<double> global_eval_times(evaluate_iter, 0.0);
    // Samples per eval slot that actually computed; see the selection logic and
    // the eval phase below.
    static std::vector<unsigned int> global_eval_samples(evaluate_iter, 0);

    // Executes one candidate and returns the algorithm the dispatch reports
    // having run. matmul_kernel_wrapper() takes `kernel` by reference and
    // rewrites it when a backend declines, so the returned value is what
    // actually executed, not what was asked for.
    auto run_algo = [&](matmul_algo_t candidate) {
        matmul_algo_t executed = candidate;
        // Go through execute_selected_algo(), not matmul_kernel_wrapper(), so a
        // candidate is measured and run exactly as it would be if the caller had
        // requested it by name -- mm partitioner included. Calling the wrapper
        // directly skipped the partitioner, so libxsmm was both timed and
        // executed in its slower unpartitioned form.
        execute_selected_algo(layout, transA == 't', transB == 't', M, N, K,
                alpha, A, lda, B, ldb, bias, beta, C, ldc, is_weights_const,
                src_type_size, out_type_size, num_threads, executed,
                lowoha_param, batch_params);
        return executed;
    };

    get_lowoha_mutex().lock();
    unsigned int iter_count = per_layer_iter_count[key_obj_auto];
    get_lowoha_mutex().unlock();

    //Skip phase: round-robin across candidates without timing. This warms up
    //every algo path before we start measuring in the eval phase,
    //so the first eval sample for each algo is not penalized by one-time setup costs.
    if (iter_count < skip_iter) {
        apilog_info("AutoTuner SKIP Iteration");
        bool computed = false;
        kernel = run_first_computing_candidate(
                get_algo(iter_count, candidates), candidates, run_algo,
                computed);

        get_lowoha_mutex().lock();
        per_layer_iter_count[key_obj_auto] = iter_count + 1;
        get_lowoha_mutex().unlock();
    }
    //Inference phase: skip + eval done, use globally chosen best algo.
    else if (iter_count >= skip_iter + evaluate_iter) {
        get_lowoha_mutex().lock();
        if (!global_best_computed) {
            // Pick the eval slot with the smallest summed time across all layers,
            // then map it back to the algo that was run at that slot via the
            // round-robin index (eval slot e ran candidates[e % num_algo]).
            const auto &times = global_eval_times;
            // Consider only slots that actually computed at least once. A slot
            // whose algorithm declined every call has a summed time of 0 and
            // would otherwise always be the minimum.
            unsigned int best_e = 0;
            bool best_found = false;
            for (unsigned int e = 0; e < times.size(); ++e) {
                if (global_eval_samples[e] == 0) { continue; }
                if (!best_found || times[e] < times[best_e]) {
                    best_e = e;
                    best_found = true;
                }
            }
            if (!best_found) {
                // Nothing computed during the whole eval phase. Keep slot 0 so
                // behaviour matches the previous code rather than leaving the
                // algo unset; the dispatch will report the failure itself.
                best_e = 0;
            }
            unsigned int best_idx = best_e % num_algo;
            global_best_algo = candidates[best_idx];
            global_best_computed = true;
            apilog_verbose("AutoTuner: selected best algo ",
                    (int32_t)global_best_algo, " (idx=", best_idx,
                    ", eval_slot=", best_e, ", total_time=", times[best_e],
                    ")");
        }
        kernel = global_best_algo;
        get_lowoha_mutex().unlock();

        bool computed = false;
        kernel = run_first_computing_candidate(
                kernel, candidates, run_algo, computed);
    }
    //Eval phase: round-robin across candidates, accumulate per-slot time globally.
    else {
        // Evaluate phase restarts round-robin from candidates[0]
        unsigned int eval_index = iter_count - skip_iter;
        unsigned int algo_idx = eval_index % num_algo;
        double cur_algo_time = 0.0;
        profiler_t profiler;

        profiler.tbp_start();

        bool computed = false;
        kernel = run_first_computing_candidate(
                candidates[algo_idx], candidates, run_algo, computed);

        profiler.tbp_stop();
        cur_algo_time = profiler.tbp_elapsedtime();
        apilog_info("AutoTuner Evaluate Iteration algo:", (int32_t)kernel,
                " time:", cur_algo_time);

        get_lowoha_mutex().lock();
        // Only accumulate a slot's time when the call actually computed. A slot
        // whose algorithm cannot run on this host would otherwise collect ~0 ms
        // and win the selection below, which compares summed times.
        if (computed) {
            global_eval_times[eval_index] += cur_algo_time;
            global_eval_samples[eval_index] += 1;
        }
        per_layer_iter_count[key_obj_auto] = iter_count + 1;
        get_lowoha_mutex().unlock();
    }
    return kernel;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
