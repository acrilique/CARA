/**
 * @file spectral_features.h
 * @brief Spectral feature extraction (STFT, windowing, filter banks).
 * 
 * This module provides core routines for computing short-time Fourier transforms,
 * generating and applying filter banks (e.g., Mel, Bark, ERB), and converting frequency
 * scales used in auditory signal processing.
 * 
 * @ingroup audio_features
 */

#ifndef SPECTRAL_FEATURES_H
    #define SPECTRAL_FEATURES_H

    #include <stddef.h>
    #include <stdbool.h>
    #include <math.h>
    #include <fftw3.h>
    #include <string.h>
    #include <omp.h>
    #include <cblas.h>

    #include "audio_io.h"

    /** @addtogroup audio_features
    *  @{
    */

    /**
    * @brief Supported filter bank types
    */
    typedef enum {
        F_MEL,       /**< Mel scale */
        F_BARK,      /**< Bark scale */
        F_ERB,       /**< Equivalent Rectangular Bandwidth */
        F_CHIRP,     /**< Chirp scale */
        F_CAM,       /**< Cambridge ERB-rate */
        F_LOG10,     /**< Log10 frequency scaling */
        F_UNKNOWN    /**< Invalid/unsupported filter type */
    } filter_type_t;

    extern const char *FILTER_TYPE_NAMES[];

    #define M_PI 3.14159265358979323846

    extern unsigned int n_threads;

    /**
    * @brief STFT data structure
    */


    typedef struct {
        double avg_fft_time_us;          // Average FFT execution time per frame
        double fft_gflops;               // GFLOPS returned by FFT_bench
        size_t num_frames;               // Total frames processed
        size_t num_threads;              // Threads used
        double parallel_efficiency;      // Parallel efficiency in %
    } stft_bench_t;


    typedef struct {
        float sample_rate;
        size_t output_size;
        size_t num_frequencies;
        size_t total_length;
        float *frequencies;
        float *magnitudes;
        float *phasers;
        stft_bench_t benchmark;
    } stft_t;

    /**
    * @brief FFT plan wrapper
    */
    typedef struct {
        fftwf_plan plan;
        float *in;
        fftwf_complex *out;
    } fft_t;

    /**
    * @brief Filter bank representation
    */
    typedef struct {
        size_t  *freq_indexs;
        float   *weights;
        size_t   size;
        size_t num_filters;
    } filter_bank_t;

    /* ---- STFT and FFT Utilities ---- */

    bool init_fft_output(stft_t *result, unsigned int window_size, unsigned int hop_size, unsigned int num_samples);
    fft_t init_fftw_plan(const size_t window_size, const char *cache_dir);
    stft_t stft(audio_data *audio, const size_t window_size, const size_t hop_size, float *window_values, fft_t *fft);
    void free_stft(stft_t *result);
    void free_fft_plan(fft_t *fft);
    void cleanup_fft_threads(fft_t *thread_ffts, const size_t num_threads);
    void calculate_frequencies(stft_t *result, size_t window_size, float sample_rate);

    void print_stft_bench(const stft_bench_t *bench);

    /* ---- Window and Filterbank Utilities ---- */

    void window_function(float *window_values, size_t window_size, const char *window_type);
    filter_bank_t gen_filterbank(filter_type_t type, float min_f, float max_f, size_t n_filters, float sr, size_t fft_size, float *filter);
    filter_type_t parse_filter_type(const char *name);
    void print_melbank(const filter_bank_t *v);

    /* ---- Frequency and Scale Utilities ---- */

    size_t hz_to_index(size_t num_freq, size_t sample_rate, float f);
    float safe_diff(size_t a, size_t b);
    float brachless_db(float mag, bool db);

    /* ---- Frequency Scale Conversion ---- */

    double hz_to_mel(double hz);     double mel_to_hz(double mel);
    double hz_to_bark(double hz);    double bark_to_hz(double bark);
    double hz_to_erb(double hz);     double erb_to_hz(double erb);
    double hz_to_chirp(double hz);   double chirp_to_hz(double chirp);
    double hz_to_cam(double hz);     double cam_to_hz(double cam);
    double hz_to_log10(double hz);   double log10_to_hz(double val);
    double hz_to_cent(double hz);    double cent_to_hz(double cent);


     #include "audio_io.h"
    #include "../utils/bench.h"

    /** @} */  // end of audio_features

#endif // SPECTRAL_FEATURES_H
