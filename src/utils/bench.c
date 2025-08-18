#include "utils/bench.h"

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

void benchmark_init() {
    benchmarks.total_time   = 0;   
    benchmarks.timing_index = 0; 
    benchmarks.start_time   = 0;
}

long long get_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

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

void format_scaled(double val, char *buffer, size_t buf_size, const char *unit) {
    const scale  s      = get_scale(val);
    const double scaled = val / s.factor;
    snprintf(buffer, buf_size, "%7.3f %s%s", scaled, s.suffix, unit);
}

double FFT_bench(double mu_s, unsigned int FFT_size, bool log) {
    if (mu_s <= 0.0)
        return 0.0;

    const double fft_points    = (double)FFT_size;
    const double time_s        = mu_s * scales[MICRO].factor; 
    const double flops         = 5.0 * fft_points * log2(fft_points); /* https://www.fftw.org/speed/ */
    const double flops_per_sec = flops / time_s;
    const double gflops        = flops_per_sec / scales[GIGA].factor; 

    return gflops;
}

int compare_times(const void *a, const void *b) {
    return ((time_info*)b)->time_us - ((time_info*)a)->time_us;
}

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

void print_bench() {
    for (size_t i = 0; i < benchmarks.timing_index; i++) {
        printf("%s:%lld\n", benchmarks.timings[i].function_name, benchmarks.timings[i].time_us);
    }
}