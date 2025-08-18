#include "audio_tools/spectral_features.h"


#define ALIGNED_ALLOC_ALIGNMENT 32


unsigned int n_threads = 1;

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

/**
 * @brief Zeroth order modified Bessel function of the first kind (I₀)
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

inline size_t hz_to_index(size_t num_freq, size_t sample_rate, float f) {
    return (size_t)((num_freq * f * 2) / sample_rate);
}

bool is_power_of_two(size_t x) {
    return x && ((x & (x - 1)) == 0);
}

void *aligned_alloc_batch(size_t alignment, size_t size, bool zero_init) {
    if (!is_power_of_two(alignment)) {
        ERROR("Alignment (%zu) must be a power of two.", alignment);
        return NULL;
    }

    void *ptr = aligned_alloc(alignment, size);
    if (!ptr) {
        ERROR("aligned_alloc failed for size %zu with alignment %zu.", size, alignment);
        return NULL;
    }

    if (zero_init) {
        memset(ptr, 0, size);
    }

    return ptr;
}

void aligned_free_batch(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

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



double hz_to_mel(double hz)       { return 2595.0 * log10(1.0 + hz / 700.0); }
double mel_to_hz(double mel)      { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

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


void print_melbank(const filter_bank_t *v) {
    if (!v || !v->weights || v->size == 0) {
        LOG("melbank is empty or NULL.");
        return;
    }

    LOG("melbank size: %zu", v->size);

    for (size_t i = 0; i < v->size;i++) 
        LOG("Index %zu -> x: %zu, Weight: %.6f", i, v->freq_indexs[i], v->weights[i]);
}


filter_bank_t gen_filterbank(filter_type_t type,
                             float min_f, float max_f,
                             size_t n_filters, float sr,
                             size_t fft_size, float *filter) {
    
    filter_bank_t non_zero = {
        .freq_indexs = NULL,
        .weights     = NULL,
        .size        = 0,
        .num_filters = n_filters
    };

    const size_t num_f     = fft_size / 2;
    const size_t avg_len   = n_filters * (fft_size * 3 / 512);
    non_zero.freq_indexs   = malloc(avg_len * sizeof(size_t));
    non_zero.weights       = malloc(avg_len * sizeof(float));
    
  
    double *bin_edges = malloc((n_filters + 2) * sizeof(double));
    if (!bin_edges) {
        ERROR("Failed to allocate bin_edges");
        return non_zero;
    }
    
    double (*hz_to_scale)(double) = NULL;
    double (*scale_to_hz)(double) = NULL;

    switch (type) {
        case F_MEL:     hz_to_scale = hz_to_mel;     scale_to_hz = mel_to_hz; break;
        case F_BARK:    hz_to_scale = hz_to_bark;    scale_to_hz = bark_to_hz; break;
        case F_ERB:     hz_to_scale = hz_to_erb;     scale_to_hz = erb_to_hz; break;
        case F_CHIRP:   hz_to_scale = hz_to_chirp;   scale_to_hz = chirp_to_hz; break;
        case F_CAM:     hz_to_scale = hz_to_cam;     scale_to_hz = cam_to_hz; break;
        case F_LOG10:   hz_to_scale = hz_to_log10;   scale_to_hz = log10_to_hz; break;
        default:
            ERROR("Unknown filterbank type enum: %d", type);
            free(bin_edges);  // Don't forget to free!
            return non_zero;
    }

    double scale_min = hz_to_scale((double)min_f);
    double scale_max = hz_to_scale((double)max_f);
    double step      = (scale_max - scale_min) / (double)(n_filters + 1);

    for (size_t i = 0; i < n_filters + 2; i++) {
        double scale = scale_min + step * (double)i;
        double hz    = scale_to_hz(scale);
        bin_edges[i] = hz * (double)(num_f) / (sr / 2.0);
    }

    for (size_t m = 1; m <= n_filters; m++) {
        double left   = bin_edges[m - 1];
        double center = bin_edges[m];
        double right  = bin_edges[m + 1];

        int k_start = (int)floor(left);
        int k_end   = (int)ceil(right);

        for (int k = k_start; k <= k_end; k++) {
            if (k < 0 || k >= (int)num_f) continue;

            double weight = 0.0;
            if (k < center) {
                weight = (k - left) / (center - left);
            } else if (k <= right) {
                weight = (right - k) / (right - center);
            } else {
                continue;
            }

            non_zero.weights[non_zero.size]     = (float)weight;
            filter[(m - 1) * num_f + k]         = (float)weight;
            non_zero.freq_indexs[non_zero.size] = k;
            non_zero.size++;
        }
    }

    free(bin_edges);
    
    non_zero.freq_indexs = realloc(non_zero.freq_indexs, non_zero.size * sizeof(size_t));
    non_zero.weights     = realloc(non_zero.weights,     non_zero.size * sizeof(float));

    return non_zero;
}


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
    result->num_frequencies = window_size / 2;
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
 * @brief Generate window coefficients
 *
 * @param window_values Pointer to pre-allocated array of size `window_size`
 * @param window_size Length of the window
 * @param window_type Name of the window (e.g., "hann", "kaiser")
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
            window_values[i] = 1.0f - fabsf((float)(i - (N - 1.0) / 2.0) / ((N - 1.0) / 2.0));
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

void calculate_frequencies(stft_t *result, size_t window_size, float sample_rate) {
    size_t half_window_size = window_size / 2;
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

stft_t stft(audio_data *audio, size_t window_size, size_t hop_size, float *window_values, fft_t *master_fft) {
    stft_t result = {0};
    
    if (!audio || !audio->samples || !window_values || !master_fft || !master_fft->plan) {
        ERROR("Invalid input: audio, window_values, or FFT plan is NULL.");
        return result;
    }

    if (audio->channels < 1 || audio->channels > 2) {
        ERROR("Invalid number of channels: %u (must be 1 or 2).", audio->channels);
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
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < length; i++) mono[i] = audio->samples[i];
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < length; i++) mono[i] = (audio->samples[i * 2] + audio->samples[i * 2 + 1]) * half;
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
        result.benchmark.fft_gflops          = FFT_bench(result.benchmark.avg_fft_time_us, window_size,true);
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

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < output_size; i++) {
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
        result.benchmark.fft_gflops           = FFT_bench(result.benchmark.avg_fft_time_us, window_size,true);

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
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < total_bins; i++) {
        float re = result.phasers[i * 2];
        float im = result.phasers[i * 2 + 1];
        result.magnitudes[i] = sqrt(re*re + im*im);
    }

    return result;
}