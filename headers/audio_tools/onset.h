/**
 * @file onset.h
 * @brief Onset detection and onset strength envelope computation
 * 
 * This module provides functions for computing onset strength envelopes,
 * which measure the rate of spectral change over time. These are fundamental
 * for beat tracking and rhythm analysis.
 * 
 * @ingroup audio_features
 */

#ifndef ONSET_H
#define ONSET_H

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "spectral_features.h"

/** @addtogroup audio_features
 *  @{
 */

/**
 * @brief Onset strength envelope result structure
 * 
 * Contains the computed onset strength values and metadata about
 * the frame rate and length.
 */
typedef struct {
    float *envelope;        /**< Onset strength values (one per frame) */
    size_t length;          /**< Number of frames */
    float frame_rate;       /**< Frames per second (sr / hop_length) */
} onset_envelope_t;

/**
 * @brief Aggregation method for combining onset across frequency bins
 */
typedef enum {
    AGG_MEAN,      /**< Arithmetic mean */
    AGG_MEDIAN     /**< Median value */
} aggregation_method_t;

/**
 * Compute onset strength envelope from a mel spectrogram.
 *
 * Onset strength at time t is determined by:
 *   mean_f max(0, S[f, t] - ref[f, t - lag])
 * 
 * where ref is S after local max filtering along the frequency axis.
 * This suppresses vibrato and other rapid frequency modulations.
 *
 * @param mel_spectrogram Input mel spectrogram in dB (n_mels × n_frames, row-major)
 * @param n_mels Number of mel frequency bands
 * @param n_frames Number of time frames
 * @param lag Time lag for computing differences (typically 1)
 * @param max_size Size (in frequency bins) of local max filter (1 = no filtering)
 * @param detrend If true, apply high-pass filter to remove DC component
 * @param aggregate Aggregation method (AGG_MEAN or AGG_MEDIAN)
 * @param ref_spec Optional pre-computed reference spectrum (can be NULL)
 * @param frame_rate Frames per second (for metadata)
 * 
 * @return Onset envelope structure with allocated envelope array
 * 
 * @note Caller is responsible for freeing the returned structure with free_onset_envelope()
 */
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
);

/**
 * Apply a local maximum filter along the frequency axis.
 *
 * For each time frame, computes the local maximum over a sliding window
 * along the frequency dimension. This is used to create a reference
 * spectrum that suppresses rapid frequency variations.
 *
 * @param input Input spectrogram (n_mels × n_frames, row-major)
 * @param output Output filtered spectrogram (same size, pre-allocated)
 * @param n_mels Number of frequency bins
 * @param n_frames Number of time frames
 * @param window_size Size of the maximum filter window
 */
void local_max_filter_1d(
    const float *input,
    float *output,
    size_t n_mels,
    size_t n_frames,
    int window_size
);

/**
 * Aggregate onset differences across frequency bins.
 *
 * Combines the per-frequency onset values into a single onset strength
 * value per time frame using either mean or median aggregation.
 *
 * @param onset_diff Input onset differences (n_mels × n_frames_out, row-major)
 * @param onset_env Output aggregated envelope (n_frames_out, pre-allocated)
 * @param n_mels Number of frequency bins
 * @param n_frames_out Number of output time frames
 * @param method Aggregation method (AGG_MEAN or AGG_MEDIAN)
 */
void aggregate_onset(
    const float *onset_diff,
    float *onset_env,
    size_t n_mels,
    size_t n_frames_out,
    aggregation_method_t method
);

/**
 * Apply simple high-pass filter to remove DC component.
 *
 * Implements a first-order IIR high-pass filter:
 *   y[n] = x[n] - 0.99 * y[n-1]
 *
 * @param signal Input/output signal (modified in-place)
 * @param length Number of samples
 */
void detrend_signal(float *signal, size_t length);

/**
 * Convert power spectrogram to decibels.
 *
 * Applies: dB = 10 * log10(power + epsilon)
 * where epsilon prevents log(0).
 *
 * @param power Input power values
 * @param db Output dB values (pre-allocated)
 * @param length Number of values
 * @param ref Reference power for normalization (use 1.0 for absolute dB)
 */
void power_to_db(const float *power, float *db, size_t length, float ref);

/**
 * Free memory allocated for an onset envelope structure.
 *
 * @param env Pointer to onset envelope structure to free
 */
void free_onset_envelope(onset_envelope_t *env);

/** @} */  // end of audio_features

#endif // ONSET_H
