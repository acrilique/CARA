#include "utils/bench.h"
#include "utils/compat.h"

/*
 * The MIT License (MIT)
 * 
 * Copyright © 2025 Devadut S Balan
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

benchmark_t benchmarks;

/**
 * Initialize global benchmark state.
 *
 * Resets the global `benchmarks` structure counters used by the benchmarking
 * utility: sets total_time, timing_index, and start_time to zero so timing can
 * begin from a clean state.
 */
void benchmark_init() {
    benchmarks.total_time   = 0;   
    benchmarks.timing_index = 0; 
    benchmarks.start_time   = 0;
}

/**
 * Return a monotonic timestamp in microseconds.
 *
 * Uses clock_gettime(CLOCK_MONOTONIC) and converts the result to microseconds
 * (seconds * 1,000,000 + nanoseconds / 1,000). The value is suitable for
 * measuring elapsed intervals but is not tied to the wall-clock calendar time.
 *
 * @return Monotonic time in microseconds as a signed 64-bit integer.
 */
#ifdef _MSC_VER
#include <windows.h>
long long get_time_us() {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (t.QuadPart * 1000000) / f.QuadPart;
}
#else
long long get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
#endif

/**
 * Record the elapsed time (since benchmarks.start_time) for a named function.
 *
 * Computes the current monotonic time, stores the elapsed microseconds and the
 * provided function name into the next slot of the global `benchmarks` object,
 * adds the elapsed time to `benchmarks.total_time`, and advances
 * `benchmarks.timing_index`.
 *
 * If the maximum number of benchmark entries (MAX_FUNS_TO_BENCH) has been
 * reached, logs an error via ERROR(...) and returns without modifying state.
 *
 * @param function_name Name of the function being recorded; copied into the
 *                      benchmark entry and nul-terminated. */
void record_timing(const char *function_name) {
    if (benchmarks.timing_index >= MAX_FUNS_TO_BENCH) {
        ERROR("Exceeded maximum number of benchmarked functions");
        return;
    }

    const long long end_time = get_time_us();
    benchmarks.timings[benchmarks.timing_index].time_us = end_time - benchmarks.start_time;
    benchmarks.total_time += benchmarks.timings[benchmarks.timing_index].time_us;

    strncpy(benchmarks.timings[benchmarks.timing_index].function_name, 
            function_name, 
            sizeof(benchmarks.timings[benchmarks.timing_index].function_name) - 1);
    
    benchmarks.timings[benchmarks.timing_index].function_name[
        sizeof(benchmarks.timings[benchmarks.timing_index].function_name) - 1] = '\0';

    benchmarks.timing_index++;
}

/**
 * Select an appropriate scale for the given numeric value.
 *
 * Determines which predefined scale yields an absolute scaled value >= 1.0 by
 * testing scales from largest to smallest. If `v` is 0.0 or no scale
 * satisfies the criterion, the function returns the `NANO` scale.
 *
 * @param v Value in the base unit to be scaled.
 * @return scale The selected scale entry (contains factor and suffix) suitable
 *         for formatting `v`. */
scale get_scale(double v) {
    if (v == 0.0) {
        return scales[NANO]; 
    }
    for (int i = SCALE_COUNT - 1; i >= 0; --i) {
        const double scaled = v / scales[i].factor;
        if (fabs(scaled) >= 1.0) {
            return scales[i];
        }
    }
    return scales[NANO]; 
}

/**
 * Format a numeric value into a human-readable string with an appropriate SI scale and unit.
 *
 * Determines an appropriate scale (e.g., k, M, μ, etc.) for 'val', scales the value, and writes
 * a formatted string of the form "<scaled> <suffix><unit>" into 'buffer'.
 *
 * @param val     Numeric value to format.
 * @param buffer  Destination buffer for the formatted string (will be null-terminated if buf_size > 0).
 * @param buf_size Size of 'buffer' in bytes; output will be truncated if it does not fit.
 * @param unit    Unit string appended after the scale suffix (e.g., "s", "B").
 */
void format_scaled(double val, char *buffer, size_t buf_size, const char *unit) {
    const scale  s      = get_scale(val);
    const double scaled = val / s.factor;
    snprintf(buffer, buf_size, "%7.3f %s%s", scaled, s.suffix, unit);
}

/**
 * Estimate FFT performance in GFLOPS for a single run.
 *
 * Given an elapsed time in microseconds and an FFT size, returns the estimated
 * gigaflops (GFLOPS) using the common FFT FLOP count approximation:
 * FLOPs ≈ 5 * N * log2(N). If mu_s <= 0.0 the function returns 0.0.
 *
 * @param mu_s Elapsed time of the FFT run in microseconds.
 * @param FFT_size Number of points in the FFT (N).
 * @param log Unused; accepted for API compatibility and ignored by this function.
 * @return Estimated performance in GFLOPS (double).
 */
double FFT_bench(double mu_s, unsigned int FFT_size) {
    if (mu_s <= 0.0)
        return 0.0;

    const double fft_points    = (double)FFT_size;
    const double time_s        = mu_s * scales[MICRO].factor; 
    const double flops         = 5.0 * fft_points * log2(fft_points); /* https://www.fftw.org/speed/ */
    const double flops_per_sec = flops / time_s;
    const double gflops        = flops_per_sec / scales[GIGA].factor; 

    return gflops;
}

/**
 * Compare two time_info entries for descending sort by elapsed microseconds.
 *
 * Interprets `a` and `b` as `time_info*` and compares their `time_us` fields.
 * Designed for use with qsort to order entries from largest to smallest `time_us`.
 *
 * @param a Pointer to the first `time_info`.
 * @param b Pointer to the second `time_info`.
 * @return Negative if `b->time_us` < `a->time_us`, zero if equal, positive if `b->time_us` > `a->time_us`.
 */
int compare_times(const void *a, const void *b) {
    return ((time_info*)b)->time_us - ((time_info*)a)->time_us;
}

/**
 * Map a percentage value to an ANSI color code string for gradient display.
 *
 * Returns a color constant representing a heat-map grade for the provided
 * percentage (expected in the 0–100 range). Thresholds are inclusive at the
 * upper bounds shown (e.g., 80.0 and above => BRIGHT_RED) and the lowest
 * nonzero range (>0.1) maps to GREEN; values <= 0.1 return BLUE.
 *
 * @param percentage Percentage value to evaluate (0.0–100.0).
 * @return Pointer to a null-terminated string constant representing the
 *         chosen color code (e.g., BRIGHT_RED, RED, MAGENTA, ...).
 */
const char* get_gradient_color(double percentage) {
    if (percentage >= 80.0) return BRIGHT_RED;   
    if (percentage >= 60.0) return RED;
    if (percentage >= 40.0) return MAGENTA;
    if (percentage >= 25.0) return BRIGHT_YELLOW;
    if (percentage >= 15.0) return YELLOW;
    if (percentage >= 5.0)  return BRIGHT_GREEN;
    if (percentage > 0.1)   return GREEN;
    return BLUE;  
}

/**
 * Print a ranked, colored table of recorded benchmark timings to stderr.
 *
 * Sorts the global `benchmarks.timings` by elapsed time (descending) and emits
 * a human-readable, colorized table showing each function's name, scaled
 * execution time, and percentage of total runtime, followed by a simple
 * ASCII bar visualization. If no timings are recorded, a warning is issued
 * and nothing is printed.
 *
 * Side effects:
 * - Writes formatted output to stderr.
 * - Reorders `benchmarks.timings` in-place via qsort.
 *
 * Notes:
 * - Times are formatted using the utility `format_scaled` (seconds shown).
 * - Color output relies on the terminal escape sequences defined in this file.
 */
void print_bench_ranked() {
    if (benchmarks.timing_index == 0) {
        WARN("No benchmark data available");
        return;
    }

    qsort(benchmarks.timings, benchmarks.timing_index, sizeof(time_info), compare_times);

    fprintf(stderr, "%s---------------------------------------------------------\n", BRIGHT_CYAN);
    fprintf(stderr, "| %-20s | %-12s | %-7s |\n", "Function", "Exec Time", "% of total runtime");
    fprintf(stderr, "---------------------------------------------------------%s\n", RESET);

    const long long max_time = benchmarks.timings[0].time_us;

    for (size_t i = 0; i < benchmarks.timing_index; i++) {
        const double percentage    = (double)benchmarks.timings[i].time_us * 100.0 / benchmarks.total_time;
        const int    filled_length = (int)(BAR_LENGTH * percentage / 100.0);

        char time_str[STRING_LENGTH];
        format_scaled(benchmarks.timings[i].time_us * scales[MICRO].factor, time_str, STRING_LENGTH, "s");

        const char *func_color = get_gradient_color((double)benchmarks.timings[i].time_us / max_time * 100.0);

        fprintf(stderr, "%s| %-20s | %12s | %6.4f%% |%s\n",
                func_color, 
                benchmarks.timings[i].function_name,
                time_str,
                percentage,
                RESET);

        fprintf(stderr, "%s[", BRIGHT_CYAN);
        for (int j = 0; j < filled_length; j++) fprintf(stderr, "▰");
        for (int j = 0; j < BAR_LENGTH - filled_length; j++) fprintf(stderr, " ");
        fprintf(stderr, "]%s\n", RESET);
    }

    fprintf(stderr, "%s---------------------------------------------------------\n%s", BRIGHT_CYAN, RESET);
}

/**
 * Print recorded benchmark entries as a JSON-like object to stdout.
 *
 * Outputs each recorded timing entry as a JSON key (the function name) with an
 * object containing:
 *   - "time_μs": elapsed time in microseconds (integer)
 *   - "percentage": percentage of the total recorded time (floating-point)
 *
 * The entire block is delimited with the markers ">>>{" and "}<<<". Entries are
 * comma-separated except for the final entry.
 */
void print_bench_json() {
    printf(">>>{\n");
    for (size_t i = 0; i < benchmarks.timing_index; i++) {
        const double percentage = (double)benchmarks.timings[i].time_us * 100.0 / benchmarks.total_time;
        printf("  \"%s\": {\"time_μs\": %lld, \"percentage\": %.2f}%s\n",
                benchmarks.timings[i].function_name,
                (long long)benchmarks.timings[i].time_us,
                percentage,
                (i < benchmarks.timing_index - 1) ? "," : "");
    }
    printf("}<<<\n");
}

/**
 * Print simple benchmark entries as "function_name:time_us" lines to stdout.
 *
 * Iterates the global `benchmarks` entries and prints each recorded timing on its
 * own line in the form "<function_name>:<time_us>" where `time_us` is the
 * elapsed time in microseconds. If no entries are recorded, nothing is printed.
 */
void print_bench() {
    for (size_t i = 0; i < benchmarks.timing_index; i++) {
        printf("%s:%lld\n", benchmarks.timings[i].function_name, benchmarks.timings[i].time_us);
    }
}
