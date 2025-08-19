#ifndef BENCH_H
    #define BENCH_H

    #include <time.h>
    #include <stdio.h>
    #include <stddef.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <stdbool.h>

    /**
    * @file bench.h
    * @brief Benchmarking and value-scaling utilities using general-purpose SI prefixes.
    * @author Devadut S Balan
    * @license MIT
    */

    #define MAX_FUNS_TO_BENCH      600     /**< Maximum number of benchmarked functions. */
    #define MAX_FUNS_NAME_LENGTH   100     /**< Maximum characters in a function label. */
    #define BAR_LENGTH             20      /**< Width of progress bars in ranking output. */

    // ─── ANSI Colors ──────────────────────────────────────────────────────────────
    #define RESET           "\x1b[0m"
    #define BRIGHT_RED      "\x1b[91m"
    #define RED             "\x1b[31m"
    #define MAGENTA         "\x1b[35m"
    #define BRIGHT_YELLOW   "\x1b[93m"
    #define YELLOW          "\x1b[33m"
    #define BRIGHT_GREEN    "\x1b[92m"
    #define GREEN           "\x1b[32m"
    #define BLUE            "\x1b[34m"
    #define BRIGHT_CYAN     "\x1b[96m"
    #define BRIGHT_BLUE     "\x1b[94m"
    #define BRIGHT_MAGENTA  "\x1b[95m"
    #define BAR_COLOR       BRIGHT_BLUE

    /**
    * @brief Unicode bar used for visual separation in terminal.
    */
    static const char line[] ="\n▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰▰\n";

    /**
    * @brief SI unit prefix scale (for time, data, size, etc.)
    */
    typedef struct {
        const char *suffix;       /**< SI unit suffix ("n", "µ", "m", "", "k", etc.) */
        double factor;     /**< Numerical divisor to scale a raw value */
    } scale;

    /**
    * @brief Enum index for each supported SI scale.
    */
    typedef enum {
            NANO        = 0,   /**< nano: 10^-9 seconds or units (1e-9) */
            MICRO       = 1,   /**< micro: 10^-6 seconds or units (1e-6) */
            MILLI       = 2,   /**< milli: 10^-3 seconds or units (1e-3) */
            UNIT        = 3,   /**< unit/base: 10^0 seconds or units (1) */
            KILO        = 4,   /**< kilo: 10^3 seconds or units (1e3) */
            MEGA        = 5,   /**< mega: 10^6 seconds or units (1e6) */
            GIGA        = 6,   /**< giga: 10^9 seconds or units (1e9) */
            TERA        = 7,   /**< tera: 10^12 seconds or units (1e12) */
            SCALE_COUNT = 8    /**< Total number of scales */
    } scale_index;

    /**
    * @brief General-purpose SI scaling table.
    *
    * These scales can be used for:
    * - Time (e.g., µs, ms)
    * - Memory (e.g., kB, MB)
    * - Frequency or rates (e.g., MFLOP/s, MHz)
    * - Any measurable quantity with power-of-ten representation
    */
    static const scale scales[SCALE_COUNT] = {
        { "n", 1e-9 },
        { "µ", 1e-6 },
        { "m", 1e-3 },
        { "",  1    },
        { "k", 1e3  },
        { "M", 1e6  },
        { "G", 1e9  },
        { "T", 1e12 }
    };

    #define STRING_LENGTH 32

    /**
    * @brief Benchmark info for a single function.
    */
    typedef struct {
        long long time_us;                            /**< Execution time in microseconds */
        char function_name[MAX_FUNS_NAME_LENGTH];     /**< Human-readable function label */
    } time_info;

    /**
    * @brief Stores timing data for all benchmarked functions.
    */
    typedef struct {
        long long  total_time;                        /**< Total time recorded */
        long long  start_time;                        /**< Internal use only */
        time_info  timings[MAX_FUNS_TO_BENCH];        /**< Per-function timing info */
        size_t     timing_index;                      /**< Number of functions tracked */
    } benchmark_t;

    /**
    * @brief Global benchmark state.
    */
    extern benchmark_t benchmarks;

    /**
    * @brief Initialize the global benchmark state.
    */
    void benchmark_init();

    /**
    * @brief Returns current high-resolution time in microseconds.
    * @return Time in µs.
    */
    long long get_time_us();

    /**
    * @brief Records the time delta since `START_TIMING()` under a named function.
    * @param function_name Label to assign to the recorded time.
    */
    void record_timing(const char *function_name);


    /**
    * @brief Compare two time_info entries (descending).
    * @param a Pointer to first time_info.
    * @param b Pointer to second time_info.
    * @return Sorting order: negative if a < b.
    */
    int compare_times(const void *a, const void *b);

    /**
    * @brief Choose the most human-readable SI scale for a value.
    * @param v Raw value (e.g., 2500000.0).
    * @return Matching scale struct with suffix and divisor.
    */
    scale get_scale(double v);

    /**
    * @brief Format a scaled value into a human-readable string.
    * @param val       Raw value to scale.
    * @param buffer    Output string buffer.
    * @param buf_size  Maximum buffer length.
    * @param unit      Unit string (e.g., "s", "FLOP/s").
    */
    void format_scaled(double val, char *buffer, size_t buf_size, const char *unit);

    /**
    * @brief Estimate MFLOP/s for a given FFT and print timing and throughput.
    * @param mu_s      Average elapsed time per FFT in microseconds.
    * @param FFT_size  Number of FFT points (N).
    */
    double FFT_bench(double mu_s, unsigned int FFT_size);

    /**
    * @brief Choose a display color for ranking output based on percentage.
    * @param percentage Time or value percentage (0.0 to 100.0)
    * @return ANSI color string.
    */
    const char* get_gradient_color(double percentage);

    /**
    * @brief Print raw timings for all benchmarked functions.
    */
    void print_bench();

    /**
    * @brief Print all benchmark data as a JSON object.
    */
    void print_bench_json();

    /**
    * @brief Pretty-print ranked benchmark chart in table and bar format.
    */
    void print_bench_ranked();

    /**
    * @brief Start timing a block of code.
    */
    #define START_TIMING()           benchmarks.start_time = get_time_us()

    /**
    * @brief Stop timing and record elapsed duration.
    * @param FUNC_NAME Label to assign this timed block.
    */
    #define END_TIMING(FUNC_NAME)    record_timing(FUNC_NAME)



/**
 * @def LOG_LEVEL
 * @brief Logging verbosity level (0 to 3).
 *
 * - 0: No logs
 * - 1: Minimal logs — only tag and message (no metadata)
 * - 2: Adds line number to the log
 * - 3: Full metadata (file, function, line)
 */
    #ifndef LOG_LEVEL
        #define LOG_LEVEL 0  /**< Default to full logs */
    #endif

    // ===== LOGGING MACROS =====

   

    #if LOG_LEVEL == 0

        /**
        * @brief Disabled LOG macro (no output at level 0).
        */
        #define LOG(...)

        /**
        * @brief Disabled WARN macro (no output at level 0).
        */
        #define WARN(...)

        /**
        * @brief Disabled ERROR macro (no output at level 0).
        */
        #define ERROR(...)

    #elif LOG_LEVEL == 1

        /**
        * @brief Log an informational message (minimal).
        * @param ... printf-style format string and optional arguments.
        * @example LOG("Initialized %d modules", count);
        */
        #define LOG(...) \
            do { \
                fprintf(stdout, BRIGHT_CYAN "[INFO] " RESET); \
                fprintf(stdout, __VA_ARGS__); \
                fprintf(stdout, "\n"); \
            } while (0)

        /**
        * @brief Log a warning message (minimal).
        * @param ... printf-style format string and optional arguments.
        * @example WARN("Unsupported mode: %s", mode);
        */
        #define WARN(...) \
            do { \
                fprintf(stderr, BRIGHT_YELLOW "[WARN] " RESET); \
                fprintf(stderr, __VA_ARGS__); \
                fprintf(stderr, "\n"); \
            } while (0)

        /**
        * @brief Log an error message (minimal).
        * @param ... printf-style format string and optional arguments.
        * @example ERROR("Failed to open file: %s", filename);
        */
        #define ERROR(...) \
            do { \
                fprintf(stderr, BRIGHT_RED "[ERROR] " RESET); \
                fprintf(stderr, __VA_ARGS__); \
                fprintf(stderr, "\n"); \
            } while (0)

    #elif LOG_LEVEL == 2

        /**
        * @brief Log an informational message with line number.
        * @param ... printf-style format string and optional arguments.
        */
        #define LOG(...) \
            do { \
                fprintf(stdout, BRIGHT_CYAN "[INFO] " RESET "[line: %d] ", __LINE__); \
                fprintf(stdout, __VA_ARGS__); \
                fprintf(stdout, "\n"); \
            } while (0)

        /**
        * @brief Log a warning message with line number.
        * @param ... printf-style format string and optional arguments.
        */
        #define WARN(...) \
            do { \
                fprintf(stderr, BRIGHT_YELLOW "[WARN] " RESET "[line: %d] ", __LINE__); \
                fprintf(stderr, __VA_ARGS__); \
                fprintf(stderr, "\n"); \
            } while (0)

        /**
        * @brief Log an error message with line number.
        * @param ... printf-style format string and optional arguments.
        */
        #define ERROR(...) \
            do { \
                fprintf(stderr, BRIGHT_RED "[ERROR] " RESET "[line: %d] ", __LINE__); \
                fprintf(stderr, __VA_ARGS__); \
                fprintf(stderr, "\n"); \
            } while (0)

    #elif LOG_LEVEL >= 3

        /**
        * @brief Log an informational message with full metadata.
        * @param ... printf-style format string and optional arguments.
        */
        #define LOG(...) \
            do { \
                fprintf(stdout, BRIGHT_CYAN "[INFO] " RESET "[file: %s | line: %d | func: %s] ", __FILE__, __LINE__, __func__); \
                fprintf(stdout, __VA_ARGS__); \
                fprintf(stdout, "\n"); \
            } while (0)

        /**
        * @brief Log a warning message with full metadata.
        * @param ... printf-style format string and optional arguments.
        */
        #define WARN(...) \
            do { \
                fprintf(stderr, BRIGHT_YELLOW "[WARN] " RESET "[file: %s | line: %d | func: %s] ", __FILE__, __LINE__, __func__); \
                fprintf(stderr, __VA_ARGS__); \
                fprintf(stderr, "\n"); \
            } while (0)

        /**
        * @brief Log an error message with full metadata.
        * @param ... printf-style format string and optional arguments.
        */
        #define ERROR(...) \
            do { \
                fprintf(stderr, BRIGHT_RED "[ERROR] " RESET "[file: %s | line: %d | func: %s] ", __FILE__, __LINE__, __func__); \
                fprintf(stderr, __VA_ARGS__); \
                fprintf(stderr, "\n"); \
            } while (0)

    #else
        #error "Invalid LOG_LEVEL. Must be 0, 1, 2, or 3."
    #endif



#endif  // BENCH_H
