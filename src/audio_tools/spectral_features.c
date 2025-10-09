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

#include "audio_tools/spectral_features.h"
#include <stdlib.h>
#ifdef _MSC_VER
#include <malloc.h>
#endif

#define ALIGNED_ALLOC_ALIGNMENT 32


unsigned int n_threads = 1;

// Slaney scale constants
static const double f_min = 0.0;
static const double f_sp = 200.0 / 3.0;
static const double min_log_hz = 1000.0;
static const double min_log_mel = (1000.0 - 0.0) / (200.0 / 3.0);
static const double logstep = 1.8562979903656262 / 27.0; // log(6.4) / 27.0

/**
 * Compute the modified Bessel function of the first kind of order zero, I0(x).
 *
 * Uses a power-series expansion I0(x) = sum_{k=0..∞} (x^2/4)^k / (k!)^2 and accumulates
 * terms until the next term is below 1e-10 times the current sum.
 *
 * @param x Input value.
 * @return Approximated I0(x).
 */
static double bessel_i0(double x) {
    double sum = 1.0;
    double y = x * x / 4.0;
    double term = y;
    int k = 1;
    while (term > 1e-10 * sum) {
        sum += term;
        k++;
        term *= y / (k * k);
    }
    return sum;
}


/**
 * Check whether an integer is a non-zero power of two.
 *
 * @param x Integer value to test.
 * @returns true if x is a power of two (and not zero), false otherwise.
 */
bool is_power_of_two(size_t x) {
    return x && ((x & (x - 1)) == 0);
}

/**
 * Allocate aligned memory with optional zero-initialization.
 *
 * Allocates a block of `size` bytes aligned to `alignment` using aligned_alloc.
 * The `alignment` must be a power of two; if not, the function returns NULL.
 * If allocation fails, NULL is returned. When `zero_init` is true, the
 * allocated memory is zeroed before being returned.
 *
 * @param alignment Byte alignment for the allocation; must be a power of two.
 * @param size Number of bytes to allocate.
 * @param zero_init If true, zero-initialize the allocated memory.
 * @return Pointer to the allocated memory on success, or NULL on failure.
 */
void *aligned_alloc_batch(size_t alignment, size_t size, bool zero_init) {
    if (!is_power_of_two(alignment)) {
        ERROR("Alignment (%zu) must be a power of two.", alignment);
        return NULL;
    }

    // Round up size to the nearest multiple of alignment
    size_t rounded_size = (size + alignment - 1) & -alignment;

#ifdef _MSC_VER
    void *ptr = _aligned_malloc(rounded_size, alignment);
#else
    void *ptr = aligned_alloc(alignment, rounded_size);
#endif
    if (!ptr) {
        ERROR("aligned_alloc failed for size %zu with alignment %zu.", rounded_size, alignment);
        return NULL;
    }

    if (zero_init) {
        memset(ptr, 0, size);
    }

    return ptr;
}

/**
 * Free memory previously allocated with aligned_alloc_batch or standard allocators.
 *
 * Safe to call with NULL; does nothing in that case.
 *
 * @param ptr Pointer to the allocated memory to free.
 */
void aligned_free_batch(void *ptr) {
    if (ptr) {
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }
}

/**
 * Print STFT benchmarking statistics to the log.
 *
 * Prints a formatted summary of the provided stft_bench_t: average FFT time per frame,
 * measured throughput in GFLOP/s, number of processed frames, number of threads used,
 * and parallel efficiency. Output is sent via the LOG macro using terminal formatting.
 *
 * @param bench Pointer to the benchmark data to print. Must be non-NULL.
 */
void print_stft_bench(const stft_bench_t *bench) {
    LOG("%s%s%s", BAR_COLOR, line, RESET);
    LOG("⏱️  %sAvg FFT time per frame%s  : %s%.3f ns%s",
           BRIGHT_CYAN, RESET, BRIGHT_YELLOW, bench->avg_fft_time_us * scales[KILO].factor, RESET);
    LOG("⚡  %sSpeed %s                 : %s%.3f%s GFLOP/S",
           BRIGHT_CYAN, RESET, BRIGHT_GREEN, bench->fft_gflops, RESET);
    LOG("📊  %sFrames processed%s       : %zu frames", BRIGHT_CYAN, RESET, bench->num_frames);
    LOG("👥  %sThreads used%s           : %zu threads", BRIGHT_CYAN, RESET, bench->num_threads);
    LOG("📈  %sParallel efficiency%s    : %.1f %%", BRIGHT_CYAN, RESET, bench->parallel_efficiency);
    LOG("%s%s%s", BAR_COLOR, line, RESET);
}


/**
 * Release FFTW resources held by an fft_t and reset its internal pointers.
 *
 * If `fft` is NULL this function does nothing. For a non-NULL `fft` it destroys
 * the FFTW plan if present and frees the input/output buffers allocated with
 * FFTW, setting those members to NULL to avoid dangling pointers.
 *
 * @param fft Pointer to the fft_t whose resources should be freed.
 */
void free_fft_plan(fft_t *fft) {
    if (fft) {
        if (fft->plan != NULL) {
            fftwf_destroy_plan(fft->plan);
            fft->plan = NULL;
        }
        if (fft->in != NULL) {
            fftwf_free(fft->in);
            fft->in = NULL;
        }
        if (fft->out != NULL) {
            fftwf_free(fft->out);
            fft->out = NULL;
        }
    }
}

/**
 * Release and clear internal buffers held by an STFT result.
 *
 * Frees aligned phasor and magnitude buffers and the frequencies array, then
 * sets those pointers in `result` to NULL. Safe to call with a NULL `result`
 * (no-op).
 *
 * @param result Pointer to an stft_t whose internal buffers will be freed. */
void free_stft(stft_t *result) {
    if (result) {
        aligned_free_batch(result->phasers);
        aligned_free_batch(result->magnitudes);
        free(result->frequencies);
        result->phasers = NULL;
        result->magnitudes = NULL;
        result->frequencies = NULL;
    }
}



/**
 * Convert frequency in Hertz to the Mel scale.
 *
 * @param hz Frequency in Hertz.
 * @param variant The Mel scale variant to use.
 * @return Corresponding value on the Mel scale.
 */
double hz_to_mel(double hz, mel_variant_t variant) {
    if (variant == MEL_SLANEY) {
        if (hz >= min_log_hz) {
            return min_log_mel + log(hz / min_log_hz) / logstep;
        } else {
            return (hz - f_min) / f_sp;
        }
    } else { // MEL_HTK
        return 2595.0 * log10(1.0 + hz / 700.0);
    }
}

/**
 * Convert a Mel-scale frequency value to linear frequency in Hertz.
 *
 * @param mel Frequency on the Mel scale.
 * @param variant The Mel scale variant to use.
 * @return Corresponding frequency in Hz.
 */
double mel_to_hz(double mel, mel_variant_t variant) {
    if (variant == MEL_SLANEY) {
        if (mel >= min_log_mel) {
            return min_log_hz * exp(logstep * (mel - min_log_mel));
        } else {
            return f_min + f_sp * mel;
        }
    } else { // MEL_HTK
        return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
    }
}

double hz_to_bark(double hz)      { return 6.0 * asinh(hz / 600.0); }
double bark_to_hz(double bark)    { return 600.0 * sinh(bark / 6.0); }

double hz_to_erb(double hz)       { return 21.4 * log10(4.37e-3 * hz + 1.0); }
double erb_to_hz(double erb)      { return (pow(10.0, erb / 21.4) - 1.0) / 4.37e-3; }

double hz_to_chirp(double hz)     { return log2(hz + 1.0); }
double chirp_to_hz(double chirp)  { return pow(2.0, chirp) - 1.0; }

double hz_to_cam(double hz)       { return 45.5 * log10(1.0 + hz / 700.0); }
double cam_to_hz(double cam)      { return 700.0 * (pow(10.0, cam / 45.5) - 1.0); }

double hz_to_log10(double hz)     { return log10(hz + 1.0); }
double log10_to_hz(double val)    { return pow(10.0, val) - 1.0; }

static mel_variant_t current_mel_variant;

static double hz_to_mel_wrapper(double hz) {
    return hz_to_mel(hz, current_mel_variant);
}

static double mel_to_hz_wrapper(double mel) {
    return mel_to_hz(mel, current_mel_variant);
}


/**
 * Print contents of a mel filter bank to the log.
 *
 * If `v` is NULL, has no weights, or `v->size` is zero, a single message is logged and the
 * function returns. Otherwise logs the bank size and one line per filter with the filter
 * index, corresponding frequency bin index (`v->freq_indexs[i]`), and filter weight
 * (`v->weights[i]`).
 *
 * @param v Pointer to the filter bank to print. */
void print_melbank(const filter_bank_t *v) {
    if (!v || !v->weights || v->size == 0) {
        LOG("melbank is empty or NULL.");
        return;
    }

    LOG("melbank size: %zu", v->size);

    for (size_t i = 0; i < v->size;i++) 
        LOG("Index %zu -> x: %zu, Weight: %.6f", i, v->freq_indexs[i], v->weights[i]);
}


/**
 * Generate a triangular filter bank in a specified frequency scale and populate the provided filter matrix.
 *
 * Builds `n_filters` triangular filters spanning [min_f, max_f] using the chosen perceptual/mathematical
 * frequency scale (mel, bark, erb, chirp, cam, or log10). The function fills the caller-provided
 * `filter` array with weights laid out as contiguous filters: filter[(m * num_bins) + k] where
 * num_bins == fft_size/2 and m ranges 0..n_filters-1. It also returns a compact representation
 * (`filter_bank_t`) containing the indices of non-zero FFT bins for all filters and their corresponding
 * weights in the same order used to populate `filter`.
 *
 * @param type      Filter bank scale/type (F_MEL, F_BARK, F_ERB, F_CHIRP, F_CAM, F_LOG10).
 * @param min_f     Lower frequency bound (Hz) of the filter bank passband.
 * @param max_f     Upper frequency bound (Hz) of the filter bank passband.
 * @param n_filters Number of triangular filters to generate.
 * @param sr        Sample rate (Hz).
 * @param fft_size  FFT size; only the first (fft_size/2) positive bins are used per filter.
 * @param filter    Caller-allocated array of length (n_filters * (fft_size/2)) to receive filter weights.
 *
 * @return A filter_bank_t whose `freq_indexs` and `weights` arrays list all non-zero bin indices and
 *         corresponding weights (in the order they were inserted). `size` is the number of non-zero entries
 *         and `num_filters` equals the requested `n_filters`. On allocation or parameter errors the returned
 *         struct will have `size == 0` and may contain NULL `freq_indexs`/`weights`.
 *
 * Notes:
 * - The caller must allocate `filter` before calling and is responsible for freeing memory owned by the
 *   returned `filter_bank_t` (freq_indexs and weights) when no longer needed.
 * - Unknown/unsupported `type` results in an error return with no constructed filters.
 */
filter_bank_t gen_filterbank(const filterbank_config_t *config, float *filter) {
    
    filter_bank_t non_zero = {
        .freq_indexs = NULL,
        .weights     = NULL,
        .size        = 0,
    };

    if (!config || config->num_filters == 0 || config->fmax <= config->fmin || config->sample_rate == 0 || config->fft_size == 0) {
        ERROR("Invalid filterbank configuration: num_filters=%zu, fmax=%.2f, fmin=%.2f, sample_rate=%zu, fft_size=%zu",
              config ? config->num_filters : 0, config ? config->fmax : 0.0, config ? config->fmin : 0.0, config ? config->sample_rate : 0, config ? config->fft_size : 0);
        return non_zero;
    }

    non_zero.num_filters = config->num_filters;

    const size_t num_f     = config->include_nyquist ? (config->fft_size / 2 + 1) : (config->fft_size / 2);
    const size_t avg_len   = config->num_filters * num_f;
    non_zero.freq_indexs   = malloc(avg_len * sizeof(size_t));
    if (!non_zero.freq_indexs) {
        ERROR("Failed to allocate freq_indexs");
        return non_zero;
    }
    non_zero.weights       = malloc(avg_len * sizeof(float));
    if (!non_zero.weights) {
        ERROR("Failed to allocate weights");
        free(non_zero.freq_indexs);
        return non_zero;
    }

    double *hz_edges = malloc((config->num_filters + 2) * sizeof(double));
    if (!hz_edges) {
        ERROR("Failed to allocate hz_edges");
        free(non_zero.freq_indexs);
        free(non_zero.weights);
        non_zero.freq_indexs = NULL;
        non_zero.weights = NULL;
        return non_zero;
    }
    
    double (*hz_to_scale)(double) = NULL;
    double (*scale_to_hz)(double) = NULL;

    current_mel_variant = config->scale_variant;

    switch (config->scale) {
        case F_MEL:
            hz_to_scale = hz_to_mel_wrapper;
            scale_to_hz = mel_to_hz_wrapper;
            break;
        case F_BARK:    hz_to_scale = hz_to_bark;    scale_to_hz = bark_to_hz; break;
        case F_ERB:     hz_to_scale = hz_to_erb;     scale_to_hz = erb_to_hz; break;
        case F_CHIRP:   hz_to_scale = hz_to_chirp;   scale_to_hz = chirp_to_hz; break;
        case F_CAM:     hz_to_scale = hz_to_cam;     scale_to_hz = cam_to_hz; break;
        case F_LOG10:   hz_to_scale = hz_to_log10;   scale_to_hz = log10_to_hz; break;
        default:
            ERROR("Unknown filterbank type enum: %d", config->scale);
            free(hz_edges);
            free(non_zero.freq_indexs);
            free(non_zero.weights);
            non_zero.freq_indexs = NULL;
            non_zero.weights     = NULL;
            return non_zero;
    }

    double scale_min, scale_max;
    if (config->scale == F_MEL) {
        scale_min = hz_to_mel(config->fmin, config->scale_variant);
        scale_max = hz_to_mel(config->fmax, config->scale_variant);
    } else {
        scale_min = hz_to_scale(config->fmin);
        scale_max = hz_to_scale(config->fmax);
    }
    
    double step = (scale_max - scale_min) / (double)(config->num_filters + 1);

    for (size_t i = 0; i < config->num_filters + 2; i++) {
        double scale = scale_min + step * (double)i;
        if (config->scale == F_MEL) {
            hz_edges[i] = mel_to_hz(scale, config->scale_variant);
        } else {
            hz_edges[i] = scale_to_hz(scale);
        }
    }

    if (config->ramp_shape == RAMP_TRIANGULAR) {
        for (size_t m = 1; m <= config->num_filters; m++) {
            double left_hz   = hz_edges[m - 1];
            double center_hz = hz_edges[m];
            double right_hz  = hz_edges[m + 1];
            
            double fdiff_left  = center_hz - left_hz;
            double fdiff_right = right_hz - center_hz;

            // Guard against division by zero
            if (fdiff_left <= 0.0 || fdiff_right <= 0.0) continue;
            
            for (size_t k = 0; k < num_f; k++) {
                double fft_freq = k * config->sample_rate / (double)config->fft_size;
                
                double ramp_left  = left_hz - fft_freq;
                double ramp_right = right_hz - fft_freq;
                
                double lower = -ramp_left / fdiff_left;
                double upper = ramp_right / fdiff_right;
                
                double weight = fmax(0.0, fmin(lower, upper));
                
                if (weight > 0.0 && non_zero.size < avg_len) {
                    non_zero.weights[non_zero.size]     = (float)weight;
                    filter[(m - 1) * num_f + k]         = (float)weight;
                    non_zero.freq_indexs[non_zero.size] = k;
                    non_zero.size++;
                }
            }
        }
    } else { // RAMP_BINWISE
        double *bin_edges = malloc((config->num_filters + 2) * sizeof(double));
        for (size_t i = 0; i < config->num_filters + 2; i++) {
            bin_edges[i] = hz_edges[i] * (double)(num_f - 1) / (config->sample_rate / 2.0);
        }

        for (size_t m = 1; m <= config->num_filters; m++) {
            double left   = bin_edges[m - 1];
            double center = bin_edges[m];
            double right  = bin_edges[m + 1];

            int k_start = (int)floor(left);
            int k_end   = (int)ceil(right);

            for (int k = k_start; k <= k_end; k++) {
                if (k < 0 || k >= (int)num_f) continue;

                double weight = 0.0;
                if (k < center) {
                    double denom = center - left;
                    if (denom > 0)
                        weight = (k - left) / denom;
                } else if (k <= right) {
                    double denom = right - center;
                    if (denom > 0)
                        weight = (right - k) / denom;
                } else {
                    continue;
                }

                if (weight > 0.0 && non_zero.size < avg_len) {
                    non_zero.weights[non_zero.size]     = (float)weight;
                    filter[(m - 1) * num_f + k]         = (float)weight;
                    non_zero.freq_indexs[non_zero.size] = k;
                    non_zero.size++;
                }
            }
        }
        free(bin_edges);
    }

    free(hz_edges);
    
    // Reallocate with temporary pointers to avoid memory leaks on failure
    size_t* temp_freq_indexs = realloc(non_zero.freq_indexs, non_zero.size * sizeof(size_t));
    if (non_zero.size > 0 && !temp_freq_indexs) {
        ERROR("Failed to reallocate freq_indexs");
        free(non_zero.freq_indexs);
        free(non_zero.weights);
        non_zero.freq_indexs = NULL;
        non_zero.weights = NULL;
        non_zero.size = 0;
        return non_zero;
    }
    non_zero.freq_indexs = temp_freq_indexs;

    float* temp_weights = realloc(non_zero.weights, non_zero.size * sizeof(float));
    if (non_zero.size > 0 && !temp_weights) {
        ERROR("Failed to reallocate weights");
        free(non_zero.freq_indexs);
        free(non_zero.weights);
        free(hz_edges);
        non_zero.freq_indexs = NULL;
        non_zero.weights = NULL;
        non_zero.size = 0;
        return non_zero;
    }
    non_zero.weights = temp_weights;

    return non_zero;
}


/**
 * Initialize STFT output buffers and metadata for an stft_t result.
 *
 * Allocates aligned arrays for magnitudes and phasors and sets frequency/count metadata
 * based on the provided window/hop sizes and number of samples. On failure any partially
 * allocated buffers are freed and the function returns false.
 *
 * @param result Pointer to an stft_t to initialize; on success its `magnitudes`,
 *               `phasers`, `frequencies`, `num_frequencies`, and `output_size` fields
 *               are populated (frequencies remains NULL here and should be computed
 *               separately).
 * @param window_size FFT window length in samples (must be > 0 and <= num_samples).
 * @param hop_size    Hop/stride between consecutive frames in samples (must be > 0).
 * @param num_samples Total number of input samples used to compute the number of frames.
 *
 * @return true if allocation and initialization succeed; false on invalid inputs or
 *         allocation failure.
 */
bool init_fft_output(stft_t *result, unsigned int window_size, unsigned int hop_size, unsigned int num_samples) {
    if (!result) {
        ERROR("Output struct is NULL.");
        return false;
    }

    if (hop_size == 0 || window_size == 0 || window_size > num_samples) {
        ERROR("Invalid window_size (%u), hop_size (%u), or window_size > num_samples (%u)",
              window_size, hop_size, num_samples);
        return false;
    }

    result->phasers         = NULL;
    result->magnitudes      = NULL;
    result->frequencies     = NULL;
    result->num_frequencies = window_size / 2 + 1;
    result->output_size     = (num_samples - window_size) / hop_size + 1;

    size_t magnitudes_size  = result->num_frequencies * result->output_size * sizeof(float);
    size_t phasers_size     = result->num_frequencies * 2 * result->output_size * sizeof(float);

    result->magnitudes      = (float*)aligned_alloc_batch(ALIGNED_ALLOC_ALIGNMENT, magnitudes_size, false);
    result->phasers         = (float*)aligned_alloc_batch(ALIGNED_ALLOC_ALIGNMENT, phasers_size, false);

    if (!result->magnitudes) {
        ERROR("Failed to allocate STFT magnitudes buffer.");
        return false;
    }

    
    if (!result->phasers) {
        ERROR("Failed to allocate STFT phasers buffer.");
        aligned_free_batch(result->magnitudes);
        result->magnitudes = NULL;
        return false;
    }

    return true;
}

/**
 * Fill a pre-allocated buffer with window coefficients.
 *
 * Generates window coefficients of the requested type and stores them in
 * `window_values[0..window_size-1]`. Supported `window_type` strings:
 * "hann", "hamming", "blackman", "blackman-harris", "bartlett",
 * "flattop", "gaussian", "kaiser". If `window_type` is unrecognized the
 * function emits a warning and writes a rectangular window (all ones).
 *
 * Requirements:
 * - `window_values` must point to an array with at least `window_size` elements.
 * - `window_size` must be greater than 1 (several window formulas divide by
 *   `window_size - 1`).
 *
 * @param window_values Buffer to receive the window coefficients (floats).
 * @param window_size Number of coefficients to generate (must be > 1).
 * @param window_type NULL-terminated name of the desired window type.
 */
void window_function(float *window_values, size_t window_size, const char *window_type) {
    const double N = (double)window_size;

    if (strcmp(window_type, "hann") == 0) {
        for (size_t i = 0; i < window_size; i++)
            window_values[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (N - 1.0f)));
    } else if (strcmp(window_type, "hamming") == 0) {
        for (size_t i = 0; i < window_size; i++)
            window_values[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (N - 1.0f));
    } else if (strcmp(window_type, "blackman") == 0) {
        for (size_t i = 0; i < window_size; i++) {
            double a0 = 0.42, a1 = 0.5, a2 = 0.08;
            double phase = 2.0 * M_PI * i / (N - 1.0);
            window_values[i] = a0 - a1 * cos(phase) + a2 * cos(2.0 * phase);
        }
    } else if (strcmp(window_type, "blackman-harris") == 0) {
        for (size_t i = 0; i < window_size; i++) {
            double phase = 2.0 * M_PI * i / (N - 1.0);
            window_values[i] = 0.35875 - 0.48829 * cos(phase) + 0.14128 * cos(2.0 * phase) - 0.01168 * cos(3.0 * phase);
        }
    } else if (strcmp(window_type, "bartlett") == 0) {
        for (size_t i = 0; i < window_size; i++)
            window_values[i] = 1.0 - fabs((i - (N - 1.0) / 2.0) / ((N - 1.0) / 2.0));
    } else if (strcmp(window_type, "flattop") == 0) {
        for (size_t i = 0; i < window_size; i++) {
            double phase = 2.0 * M_PI * i / (N - 1.0);
            window_values[i] = 1.0 - 1.93 * cos(phase) + 1.29 * cos(2.0 * phase) - 0.388 * cos(3.0 * phase) + 0.028 * cos(4.0 * phase);
        }
    } else if (strcmp(window_type, "gaussian") == 0) {
        double sigma = 0.4;
        double denom = sigma * (N - 1.0) / 2.0;
        for (size_t i = 0; i < window_size; i++) {
            double x = (i - (N - 1.0) / 2.0) / denom;
            window_values[i] = exp(-0.5 * x * x);
        }
    } else if (strcmp(window_type, "kaiser") == 0) {
        double alpha = 3.0;
        double denom = bessel_i0(alpha);
        for (size_t i = 0; i < window_size; i++) {
            double ratio = 2.0 * i / (N - 1.0) - 1.0;
            window_values[i] = (float)(bessel_i0(alpha * sqrt(1.0 - ratio * ratio)) / denom);
        }
    } else {
        WARN("Unknown window type: %s. Using rectangular window.", window_type);
        for (size_t i = 0; i < window_size; i++)
            window_values[i] = 1.0f;
    }
}

/**
 * Compute and store FFT bin center frequencies for the positive half of the spectrum.
 *
 * Allocates and fills result->frequencies with (window_size/2) entries where
 * frequencies[i] = i * (sample_rate / window_size). The array is heap-allocated
 * and the caller takes ownership (free with free_stft or aligned_free_batch as appropriate).
 * If allocation fails, result->frequencies remains NULL and an error is logged.
 *
 * @param result Pointer to the stft_t whose `frequencies` field will be set.
 * @param window_size FFT window size; only the positive half (window_size/2) is generated.
 * @param sample_rate Sample rate in Hz used to convert bin indices to frequencies.
 */
void calculate_frequencies(stft_t *result, size_t window_size, float sample_rate) {
    size_t half_window_size = window_size / 2 + 1;
    float scale = sample_rate / window_size;
    result->frequencies = (float*)malloc(half_window_size * sizeof(float));
    if (!result->frequencies) {
        ERROR("Failed to allocate frequency array.");
        return;
    }
    for (size_t i = 0; i < half_window_size; i++) {
        result->frequencies[i] = i * scale;
    }
}

/**
 * Initialize an FFTW real-to-complex plan and allocate aligned I/O buffers.
 *
 * Attempts to load FFTW "wisdom" from a file named "<cache_dir>/<window_size>.wisdom" to create
 * an optimized plan. If wisdom is unavailable or fails to import, creates a measured plan and
 * attempts to save wisdom back to the same file. Allocates input (float[]) of length `window_size`
 * and output (fftwf_complex[]) of length `window_size/2 + 1`.
 *
 * @param window_size Length of the real input frame (number of samples / FFT size).
 * @param cache_dir Directory path used to read/write FFTW wisdom files (must be non-NULL and reasonably short).
 * @return An initialized fft_t structure containing allocated `in` and `out` buffers and `plan` on success.
 *         On failure (invalid input, allocation failure, or plan creation failure) returns a zeroed fft_t
 *         with NULL pointers for `in`, `out`, and `plan`.
 */
fft_t init_fftw_plan(const size_t window_size, const char *cache_dir) {
    fft_t fft = {0};
    if (!cache_dir || strlen(cache_dir) > 200) {
        ERROR("Invalid or too long cache_dir.");
        return fft;
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/%zu.wisdom", cache_dir, window_size);

    fft.in          = (float *)fftwf_malloc(sizeof(float) * window_size);
    fft.out         = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (window_size / 2 + 1));

    if (!fft.in || !fft.out) {
        ERROR("Memory allocation failed for FFTW buffers.");
        fftwf_free(fft.in);
        fftwf_free(fft.out);
        fft.in = NULL;
        fft.out = NULL;
        return fft;
    }

    FILE *wisdom_file = fopen(filename, "r");

    if (wisdom_file) {
        if (fftwf_import_wisdom_from_file(wisdom_file)) {
            LOG("Loaded optimized FFT plan: %s", filename);
            fft.plan = fftwf_plan_dft_r2c_1d((int)window_size, fft.in, fft.out, FFTW_WISDOM_ONLY);
        } else {
            ERROR("Error importing wisdom from file: %s", filename);
        }
        fclose(wisdom_file);
    }

    if (fft.plan == NULL) {
        LOG("Cache not found or import failed. Creating FFT plan...");
        fft.plan = fftwf_plan_dft_r2c_1d((int)window_size, fft.in, fft.out, FFTW_MEASURE);

        FILE *wisdom_out = fopen(filename, "w");
        if (wisdom_out) {
            fftwf_export_wisdom_to_file(wisdom_out);
            LOG("Saved optimized FFT plan: %s", filename);
            fclose(wisdom_out);
        } else {
            ERROR("Error saving wisdom to file: %s", filename);
        }
    }

    return fft;
}

/**
 * Compute the Short-Time Fourier Transform (STFT) of an audio buffer.
 *
 * Processes `audio->samples` into an stft_t containing phasors, magnitudes,
 * frequency bins, and timing/benchmark information. The function supports
 * single-threaded execution (uses `master_fft`) and a multi-threaded path
 * when the global `n_threads` > 1 (allocates per-thread FFT buffers and plans).
 *
 * On error (invalid input or allocation/initialization failure) a zero-initialized
 * stft_t is returned.
 *
 * @param audio Pointer to input audio_data (must contain valid samples, sample_rate,
 *              num_samples and channels = 1 or 2). Channels will be converted to mono
 *              by copying (1) or averaging stereo pairs (2).
 * @param window_size Number of samples in each analysis window (FFT length).
 * @param hop_size Number of samples between successive frames (hop/stride).
 * @param window_values Pointer to an array of `window_size` window coefficients.
 * @param master_fft Pointer to a prepared fft_t providing a master FFT plan and
 *                   optional wisdom; used directly in the single-threaded path and
 *                   consulted when creating per-thread plans in the multi-threaded path.
 *
 * @return A populated stft_t on success (contains `phasers`, `magnitudes`, `frequencies`
 *         and `benchmark` fields). On failure a zero-initialized stft_t is returned.
 */
stft_t stft(audio_data *audio, size_t window_size, size_t hop_size, float *window_values, fft_t *master_fft) {
    stft_t result = {0};
    
    if (!audio || !audio->samples || !window_values || !master_fft || !master_fft->plan) {
        ERROR("Invalid input: audio, window_values, or FFT plan is NULL.");
        return result;
    }

    if (audio->channels < 1 || audio->channels > 2) {
        ERROR("Invalid number of channels: %zu (must be 1 or 2).", audio->channels);
        return result;
    }

    result.sample_rate = (float)audio->sample_rate;

   
    
    audio->channels              = (audio->channels != 0) ? audio->channels : 1;
    const size_t channels        = audio->channels;
    const size_t length          = audio->num_samples / channels;



    if (!init_fft_output(&result, window_size, hop_size, length)) {
        ERROR("Failed to initialize STFT output.");
        return result;
    }

    calculate_frequencies(&result, window_size, audio->sample_rate);

    if (!result.frequencies) {
        ERROR("Failed to calculate frequencies.");
        free_stft(&result);
        return result;
    }

    const size_t output_size      = result.output_size;
    const size_t num_frequencies  = result.num_frequencies;
    const size_t complex_size     = window_size * sizeof(float);
    const size_t output_copy_size = num_frequencies * sizeof(fftwf_complex);
    const float half              = 0.5f;


    float *mono = (float *)fftwf_malloc(length * sizeof(float));
    if (!mono) {
        ERROR("Memory allocation failed for mono buffer.");
        free_stft(&result);
        return result;
    }
    memset(mono, 0, length * sizeof(float));

    if (channels == 1) {
        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < length; i++) mono[i] = audio->samples[i];
    } else {
        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < length; i++) mono[i] = (audio->samples[i * 2] + audio->samples[i * 2 + 1]) * half;
    }

    long long total_fft_time = 0;
    long long start_time, end_time;

    if(n_threads == 1){
        fftwf_plan plan    = master_fft->plan;
        float *in          = master_fft->in;
        fftwf_complex *out = master_fft->out;

        start_time = get_time_us();
        for (size_t i = 0; i < output_size; i++) {
            size_t start_idx = i * hop_size;
            for (size_t j = 0; j < window_size; j++) in[j] = mono[start_idx + j] * window_values[j];
            long long fft_start = get_time_us();
            fftwf_execute(plan);
            total_fft_time += (get_time_us() - fft_start);
            memcpy(&(result.phasers[i * num_frequencies * 2]), out, output_copy_size);
        }
        end_time = get_time_us();

        result.benchmark.num_threads         = 1;
        result.benchmark.num_frames          = output_size;
        result.benchmark.avg_fft_time_us     = (double)total_fft_time / output_size;
        result.benchmark.parallel_efficiency = 100.0;
        result.benchmark.fft_gflops          = FFT_bench(result.benchmark.avg_fft_time_us, window_size);
    }
    else{
        LOG("function got thrds = %d", n_threads);
        omp_set_num_threads(n_threads);
        float **in_array                     = (float **)malloc(n_threads * sizeof(float *));
        fftwf_complex **out_array            = (fftwf_complex **)malloc(n_threads * sizeof(fftwf_complex *));
        fftwf_plan *plan_array               = (fftwf_plan *)malloc(n_threads * sizeof(fftwf_plan));
        long long *thread_fft_times          = (long long *)calloc(n_threads, sizeof(long long));

        for (size_t t = 0; t < n_threads; t++) {
            in_array[t] = (float *)fftwf_malloc(complex_size);
            out_array[t] = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * (window_size / 2 + 1));
            plan_array[t] = fftwf_plan_dft_r2c_1d(window_size, in_array[t], out_array[t], master_fft->plan ? FFTW_WISDOM_ONLY : FFTW_MEASURE);
        }

        start_time = get_time_us();

        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < output_size; i++) {
            int tid = omp_get_thread_num();
            float *in_local          = in_array[tid];
            fftwf_complex *out_local = out_array[tid];
            fftwf_plan plan_local    = plan_array[tid];
            size_t start_idx         = i * hop_size;

            for (size_t j = 0; j < window_size; j++)
                in_local[j] = mono[start_idx + j] * window_values[j];

            long long fft_start      = get_time_us();
            fftwf_execute(plan_local);

            thread_fft_times[tid]    += (get_time_us() - fft_start);
            memcpy(&result.phasers[i * num_frequencies * 2], out_local, output_copy_size);
        }
        end_time = get_time_us();

        total_fft_time = 0;
        for (size_t t = 0; t < n_threads; t++) total_fft_time += thread_fft_times[t];

        result.benchmark.num_threads          = n_threads;
        result.benchmark.num_frames           = output_size;
        result.benchmark.avg_fft_time_us      = (double)((total_fft_time/(double)n_threads)  / output_size);
        result.benchmark.parallel_efficiency  = (total_fft_time / (double)n_threads * 100.0) / (end_time - start_time);
        result.benchmark.fft_gflops           = FFT_bench(result.benchmark.avg_fft_time_us, window_size);

        for (size_t t = 0; t < n_threads; t++) {
            fftwf_destroy_plan(plan_array[t]);
            fftwf_free(in_array[t]);
            fftwf_free(out_array[t]);
        }

        free(plan_array);
        free(in_array);
        free(out_array);
        free(thread_fft_times);
    }
        

    fftwf_free(mono);

    const size_t total_bins = output_size * num_frequencies;
    #if defined(_MSC_VER)
        int i;
        #pragma omp parallel for schedule(static)
        for (i = 0; i < total_bins; i++) {
            float re = result.phasers[i * 2];
            float im = result.phasers[i * 2 + 1];
            result.magnitudes[i] = sqrtf(re*re + im*im);
        }
    #else
        #pragma omp parallel for simd schedule(static)
        for (int i = 0; i < total_bins; i++) {
            float re = result.phasers[i * 2];
            float im = result.phasers[i * 2 + 1];
            result.magnitudes[i] = sqrtf(re*re + im*im);
        }
    #endif

    return result;
}
