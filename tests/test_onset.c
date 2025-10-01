/**
 * @file test_onset.c
 * @brief Test program for onset strength detection
 * 
 * This program tests the onset detection module by:
 * 1. Loading a test audio file
 * 2. Computing STFT and mel spectrogram
 * 3. Computing onset strength envelope
 * 4. Saving results for comparison with librosa
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_tools/audio_visualizer.h"
#include "audio_tools/onset.h"
#include "utils/bench.h"

/**
 * Save onset envelope to a text file for comparison with librosa.
 */
static void save_onset_envelope(const char *filename, const onset_envelope_t *env) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        ERROR("Failed to open file for writing: %s", filename);
        return;
    }
    
    fprintf(fp, "# Onset strength envelope\n");
    fprintf(fp, "# Length: %zu frames\n", env->length);
    fprintf(fp, "# Frame rate: %.2f Hz\n", env->frame_rate);
    fprintf(fp, "# Format: frame_index onset_strength\n");
    
    for (size_t i = 0; i < env->length; i++) {
        fprintf(fp, "%zu %.8f\n", i, env->envelope[i]);
    }
    
    fclose(fp);
    LOG("Saved onset envelope to: %s", filename);
}

/**
 * Print statistics about the onset envelope.
 */
static void print_onset_stats(const onset_envelope_t *env) {
    if (!env || !env->envelope || env->length == 0) {
        LOG("Empty onset envelope");
        return;
    }
    
    float min_val = env->envelope[0];
    float max_val = env->envelope[0];
    float sum = 0.0f;
    
    for (size_t i = 0; i < env->length; i++) {
        float val = env->envelope[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    
    float mean = sum / env->length;
    
    LOG("%s%s%s", BAR_COLOR, line, RESET);
    LOG("📊  %sOnset Envelope Statistics%s", BRIGHT_CYAN, RESET);
    LOG("%s%s%s", BAR_COLOR, line, RESET);
    LOG("  Length:     %zu frames", env->length);
    LOG("  Frame rate: %.2f Hz", env->frame_rate);
    LOG("  Min value:  %.6f", min_val);
    LOG("  Max value:  %.6f", max_val);
    LOG("  Mean value: %.6f", mean);
    LOG("%s%s%s", BAR_COLOR, line, RESET);
}

int main(int argc, char *argv[]) {
    const char *input_file = "tests/files/black_woodpecker.wav";
    const char *output_file = "outputs/onset_envelope.txt";
    
    // Allow custom input file
    if (argc > 1) {
        input_file = argv[1];
    }
    if (argc > 2) {
        output_file = argv[2];
    }
    
    LOG("%s=== Onset Strength Detection Test ===%s", BRIGHT_CYAN, RESET);
    LOG("Input file: %s", input_file);
    LOG("Output file: %s", output_file);
    
    // Step 1: Load audio
    START_TIMING();
    audio_data audio = auto_detect(input_file);
    END_TIMING("audio_load");
    
    if (!audio.samples || audio.num_samples == 0) {
        ERROR("Failed to load audio file: %s", input_file);
        return 1;
    }
    
    LOG("Loaded audio: %zu samples, %zu Hz, %zu channels",
        audio.num_samples, audio.sample_rate, audio.channels);
    
    // Step 2: Compute STFT
    const size_t window_size = 2048;
    const size_t hop_size = 512;
    const size_t n_mels = 128;
    
    float *window_values = (float *)malloc(window_size * sizeof(float));
    window_function(window_values, window_size, "hann");
    
    START_TIMING();
    fft_t fft_plan = init_fftw_plan(window_size, "cache/FFT");
    END_TIMING("fft_init");
    
    START_TIMING();
    stft_t stft_result = stft(&audio, window_size, hop_size, window_values, &fft_plan);
    END_TIMING("stft");
    
    if (!stft_result.magnitudes) {
        ERROR("STFT computation failed");
        free(window_values);
        free_audio(&audio);
        return 1;
    }
    
    LOG("STFT computed: %zu frequencies × %zu frames",
        stft_result.num_frequencies, stft_result.output_size);
    
    // Step 3: Compute mel spectrogram
    float *filterbank = (float *)calloc((stft_result.num_frequencies + 1) * (n_mels + 2), sizeof(float));
    
    START_TIMING();
    filter_bank_t bank = gen_filterbank(F_MEL, 0.0f, audio.sample_rate / 2.0f,
                                       n_mels, audio.sample_rate, window_size, filterbank);
    END_TIMING("mel_filterbank");
    
    bounds2d_t bounds = {0};
    bounds.time.start_d = 0;
    bounds.time.end_d = stft_result.output_size;
    bounds.freq.start_d = 0;
    bounds.freq.end_d = stft_result.num_frequencies;

    const size_t t_len = stft_result.output_size;
    const size_t f_len = stft_result.num_frequencies;
    float *stft_power = (float *)malloc(t_len * f_len * sizeof(float));

    // Convert magnitude to power (magnitude² = power) to match librosa
    START_TIMING();
    for (size_t i = 0; i < t_len * f_len; i++) {
        stft_power[i] = stft_result.magnitudes[i] * stft_result.magnitudes[i];
    }
    END_TIMING("mag_to_power");
    
    START_TIMING();
    float *mel_spec_time_major = apply_filter_bank(stft_power, n_mels, f_len, filterbank, &bounds);
    END_TIMING("mel_apply");
    
    if (!mel_spec_time_major) {
        ERROR("Mel spectrogram computation failed");
        free(stft_power);
        free(filterbank);
        free(window_values);
        free_stft(&stft_result);
        free_audio(&audio);
        return 1;
    }
    
    // Transpose mel_spec from (time × mels) to (mels × time) for onset_strength
    float *mel_spec = (float *)malloc(n_mels * t_len * sizeof(float));
    if (!mel_spec) {
        ERROR("Failed to allocate transposed mel spectrogram");
        free(mel_spec_time_major);
        free(stft_power);
        free(filterbank);
        free(window_values);
        free_stft(&stft_result);
        free_audio(&audio);
        return 1;
    }
    
    for (size_t t = 0; t < t_len; t++) {
        for (size_t m = 0; m < n_mels; m++) {
            // Use absolute value to handle numerical errors from BLAS
            float val = mel_spec_time_major[t * n_mels + m];
            mel_spec[m * t_len + t] = fabsf(val);
        }
    }
    free(mel_spec_time_major);
    
    LOG("Mel spectrogram computed: %zu mels × %zu frames", n_mels, t_len);
    
    // Step 4: Convert to dB (librosa uses power_to_db with ref=np.max)
    START_TIMING();
    float *mel_db = (float *)malloc(n_mels * t_len * sizeof(float));
    
    // Find max value for reference
    float max_val = mel_spec[0];
    for (size_t i = 1; i < n_mels * t_len; i++) {
        if (mel_spec[i] > max_val) max_val = mel_spec[i];
    }
    
    power_to_db(mel_spec, mel_db, n_mels * t_len, max_val);
    END_TIMING("power_to_db");
    
    // Step 5: Compute onset strength with different parameters
    float frame_rate = (float)audio.sample_rate / hop_size;
    
    LOG("\n%s=== Testing Onset Strength (Mean Aggregation) ===%s", BRIGHT_YELLOW, RESET);
    START_TIMING();
    onset_envelope_t onset_mean = onset_strength(
        mel_db, n_mels, t_len,
        1,           // lag
        1,           // max_size (no filtering)
        false,       // detrend
        AGG_MEAN,    // aggregation
        NULL,        // ref_spec
        frame_rate
    );
    END_TIMING("onset_mean");
    
    if (onset_mean.envelope) {
        print_onset_stats(&onset_mean);
        
        char output_mean[256];
        snprintf(output_mean, sizeof(output_mean), "%s_mean.txt", output_file);
        save_onset_envelope(output_mean, &onset_mean);
    }
    
    LOG("\n%s=== Testing Onset Strength (Median Aggregation) ===%s", BRIGHT_YELLOW, RESET);
    START_TIMING();
    onset_envelope_t onset_median = onset_strength(
        mel_db, n_mels, t_len,
        1,           // lag
        1,           // max_size
        false,       // detrend
        AGG_MEDIAN,  // aggregation
        NULL,        // ref_spec
        frame_rate
    );
    END_TIMING("onset_median");
    
    if (onset_median.envelope) {
        print_onset_stats(&onset_median);
        save_onset_envelope(output_file, &onset_median);
    }
    
    LOG("\n%s=== Testing with Local Max Filter ===%s", BRIGHT_YELLOW, RESET);
    START_TIMING();
    onset_envelope_t onset_filtered = onset_strength(
        mel_db, n_mels, t_len,
        1,           // lag
        3,           // max_size (local max filter)
        false,       // detrend
        AGG_MEDIAN,  // aggregation
        NULL,        // ref_spec
        frame_rate
    );
    END_TIMING("onset_filtered");
    
    if (onset_filtered.envelope) {
        print_onset_stats(&onset_filtered);
        
        char output_filtered[256];
        snprintf(output_filtered, sizeof(output_filtered), "%s_filtered.txt", output_file);
        save_onset_envelope(output_filtered, &onset_filtered);
    }
    
    LOG("\n%s=== Testing with Detrending ===%s", BRIGHT_YELLOW, RESET);
    START_TIMING();
    onset_envelope_t onset_detrend = onset_strength(
        mel_db, n_mels, t_len,
        1,           // lag
        1,           // max_size
        true,        // detrend
        AGG_MEDIAN,  // aggregation
        NULL,        // ref_spec
        frame_rate
    );
    END_TIMING("onset_detrend");
    
    if (onset_detrend.envelope) {
        print_onset_stats(&onset_detrend);
        
        char output_detrend[256];
        snprintf(output_detrend, sizeof(output_detrend), "%s_detrend.txt", output_file);
        save_onset_envelope(output_detrend, &onset_detrend);
    }
    
    // Print benchmark summary
    LOG("\n%s=== Benchmark Summary ===%s", BRIGHT_GREEN, RESET);
    print_bench_ranked();
    
    // Cleanup
    free_onset_envelope(&onset_mean);
    free_onset_envelope(&onset_median);
    free_onset_envelope(&onset_filtered);
    free_onset_envelope(&onset_detrend);
    free(mel_db);
    free(mel_spec);
    free(stft_power);
    free(filterbank);
    free(bank.freq_indexs);
    free(bank.weights);
    free(window_values);
    free_stft(&stft_result);
    free_fft_plan(&fft_plan);
    free_audio(&audio);
    
    LOG("\n%s✅ Test completed successfully!%s", BRIGHT_GREEN, RESET);
    return 0;
}
