#include "audio_tools/onset.h"
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

/**
 * Comparison function for qsort (ascending order).
 */
static int float_compare(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

void power_to_db(const float *power, float *db, size_t length, float ref) {
    const float epsilon = 1e-10f;
    const float scale = 10.0f / logf(10.0f);  // Convert to log10
    
    if (ref <= 0.0f) ref = 1.0f;
    
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < length; i++) {
        float p = power[i] / ref;
        db[i] = scale * logf(p + epsilon);
    }
}

void local_max_filter_1d(
    const float *input,
    float *output,
    size_t n_mels,
    size_t n_frames,
    int window_size
) {
    if (window_size <= 1) {
        // No filtering needed
        memcpy(output, input, n_mels * n_frames * sizeof(float));
        return;
    }
    
    int half_window = window_size / 2;
    
    // Process each time frame independently
    #pragma omp parallel for schedule(static)
    for (size_t t = 0; t < n_frames; t++) {
        // For each frequency bin in this frame
        for (size_t f = 0; f < n_mels; f++) {
            float max_val = -INFINITY;
            
            // Find max in window [f - half_window, f + half_window]
            int start = (int)f - half_window;
            int end = (int)f + half_window;
            
            if (start < 0) start = 0;
            if (end >= (int)n_mels) end = (int)n_mels - 1;
            
            for (int k = start; k <= end; k++) {
                float val = input[k * n_frames + t];
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            output[f * n_frames + t] = max_val;
        }
    }
}

void aggregate_onset(
    const float *onset_diff,
    float *onset_env,
    size_t n_mels,
    size_t n_frames_out,
    aggregation_method_t method
) {
    if (method == AGG_MEAN) {
        // Compute mean across frequency bins for each time frame
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < n_frames_out; t++) {
            float sum = 0.0f;
            for (size_t f = 0; f < n_mels; f++) {
                sum += onset_diff[f * n_frames_out + t];
            }
            onset_env[t] = sum / (float)n_mels;
        }
    } else if (method == AGG_MEDIAN) {
        // Compute median across frequency bins for each time frame
        // Note: Librosa's np.median includes zeros, so we do the same
        float *temp = (float *)malloc(n_mels * sizeof(float));
        if (!temp) {
            ERROR("Failed to allocate temporary buffer for median computation");
            return;
        }
        
        for (size_t t = 0; t < n_frames_out; t++) {
            // Copy column
            for (size_t f = 0; f < n_mels; f++) {
                temp[f] = onset_diff[f * n_frames_out + t];
            }
            
            // Sort and find median (including zeros, matching librosa)
            qsort(temp, n_mels, sizeof(float), float_compare);
            
            if (n_mels % 2 == 0) {
                // Even number: average of two middle elements
                onset_env[t] = (temp[n_mels / 2 - 1] + temp[n_mels / 2]) * 0.5f;
            } else {
                // Odd number: middle element
                onset_env[t] = temp[n_mels / 2];
            }
        }
        
        free(temp);
    }
}

void detrend_signal(float *signal, size_t length) {
    if (length == 0) return;
    
    // First-order IIR high-pass filter: y[n] = x[n] - 0.99 * y[n-1]
    const float alpha = 0.99f;
    float prev = 0.0f;
    
    for (size_t i = 0; i < length; i++) {
        float curr = signal[i] - alpha * prev;
        signal[i] = curr;
        prev = curr;
    }
}

onset_envelope_t onset_strength(
    const float *mel_spectrogram,
    size_t n_mels,
    size_t n_frames,
    int lag,
    int max_size,
    bool detrend,
    aggregation_method_t aggregate,
    float *ref_spec,
    float frame_rate
) {
    onset_envelope_t result = {0};
    
    if (!mel_spectrogram || n_mels == 0 || n_frames == 0) {
        ERROR("Invalid input parameters to onset_strength");
        return result;
    }
    
    if (lag < 1) {
        WARN("lag must be >= 1, setting to 1");
        lag = 1;
    }
    
    if (max_size < 1) {
        WARN("max_size must be >= 1, setting to 1");
        max_size = 1;
    }
    
    if ((size_t)lag >= n_frames) {
        ERROR("lag (%d) must be less than n_frames (%zu)", lag, n_frames);
        return result;
    }
    
    // Step 1: Compute or use provided reference spectrum
    bool allocated_ref = false;
    if (ref_spec == NULL) {
        ref_spec = (float *)malloc(n_mels * n_frames * sizeof(float));
        if (!ref_spec) {
            ERROR("Failed to allocate reference spectrum");
            return result;
        }
        allocated_ref = true;
        
        if (max_size == 1) {
            // No filtering, just copy
            memcpy(ref_spec, mel_spectrogram, n_mels * n_frames * sizeof(float));
        } else {
            // Apply local max filter
            local_max_filter_1d(mel_spectrogram, ref_spec, n_mels, n_frames, max_size);
        }
    }
    
    // Step 2: Compute temporal differences with lag
    size_t n_frames_out = n_frames - lag;
    float *onset_diff = (float *)malloc(n_mels * n_frames_out * sizeof(float));
    if (!onset_diff) {
        ERROR("Failed to allocate onset difference buffer");
        if (allocated_ref) free(ref_spec);
        return result;
    }
    
    // Compute: max(0, S[f, t] - ref[f, t - lag])
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t f = 0; f < n_mels; f++) {
        for (size_t t = 0; t < n_frames_out; t++) {
            size_t curr_idx = f * n_frames + (t + lag);
            size_t prev_idx = f * n_frames + t;
            
            float diff = mel_spectrogram[curr_idx] - ref_spec[prev_idx];
            onset_diff[f * n_frames_out + t] = fmaxf(0.0f, diff);
        }
    }
    
    // Step 3: Aggregate across frequency bins
    float *onset_env = (float *)malloc(n_frames_out * sizeof(float));
    if (!onset_env) {
        ERROR("Failed to allocate onset envelope buffer");
        free(onset_diff);
        if (allocated_ref) free(ref_spec);
        return result;
    }
    
    aggregate_onset(onset_diff, onset_env, n_mels, n_frames_out, aggregate);
    
    // Step 4: Pad to match original length (prepend zeros for lag)
    float *padded = (float *)calloc(n_frames, sizeof(float));
    if (!padded) {
        ERROR("Failed to allocate padded envelope buffer");
        free(onset_env);
        free(onset_diff);
        if (allocated_ref) free(ref_spec);
        return result;
    }
    
    memcpy(padded + lag, onset_env, n_frames_out * sizeof(float));
    free(onset_env);
    
    // Step 5: Optional detrending
    if (detrend) {
        detrend_signal(padded, n_frames);
    }
    
    // Step 6: Populate result structure
    result.envelope = padded;
    result.length = n_frames;
    result.frame_rate = frame_rate;
    
    // Cleanup
    free(onset_diff);
    if (allocated_ref) {
        free(ref_spec);
    }
    
    return result;
}

void free_onset_envelope(onset_envelope_t *env) {
    if (env) {
        if (env->envelope) {
            free(env->envelope);
            env->envelope = NULL;
        }
        env->length = 0;
        env->frame_rate = 0.0f;
    }
}
