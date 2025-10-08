#include "audio_tools/audio_visualizer.h"
#include "utils/bench.h"  // Add this include for logging macros

/**
 * Convert a magnitude to either linear or a compact dB-like value.
 *
 * If `db` is false, returns the input magnitude unchanged. If `db` is true,
 * returns a bounded dB-like measure computed as log10(1 + mag^2), which
 * emphasizes larger magnitudes while remaining finite for mag = 0.
 *
 * @param mag Linear magnitude value.
 * @param db If true, return the dB-like transformed value; if false, return `mag`.
 * @return The original magnitude or its dB-like transformation.
 */


inline float brachless_db(float mag, bool db) {
    return !db * mag + db * log10f(1 + mag * mag);
}

/**
 * Log the current time and frequency bounds from a bounds2d_t.
 *
 * Prints both the floating-point ("f") and discrete/index ("d") start and end
 * values for time and frequency to the application's logging facility.
 *
 * @param bounds Pointer to the bounds2d_t whose fields will be read and logged.
 */
void print_bounds(bounds2d_t *bounds) {
    LOG("Time bounds:");
    LOG("  Start (f): %.2f", bounds->time.start_f);
    LOG("  End   (f): %.2f", bounds->time.end_f);
    LOG("  Start (d): %zu", bounds->time.start_d);
    LOG("  End   (d): %zu", bounds->time.end_d);

    LOG("Frequency bounds:");
    LOG("  Start (f): %.2f", bounds->freq.start_f);
    LOG("  End   (f): %.2f", bounds->freq.end_f);
    LOG("  Start (d): %zu", bounds->freq.start_d);
    LOG("  End   (d): %zu", bounds->freq.end_d);
}

/**
 * Update the time upper bound of a bounds2d_t.
 *
 * Sets bounds->time.end_d to max_time. The max_freq parameter is unused and retained for API compatibility.
 *
 * @param bounds Bounds structure to modify.
 * @param max_freq Unused.
 * @param max_time New time upper bound to assign to bounds->time.end_d.
 */
void set_limits(bounds2d_t *bounds, const size_t max_freq, const size_t max_time) {
    bounds->time.end_d = max_time;
    bounds->freq.end_d = max_freq;
}

/**
 * Initialize 2D bounds' digital indices from an STFT result.
 *
 * Converts the frequency bounds' floating-point Hz values in `bounds->freq`
 * to discrete bin indices using the STFT's number of frequency bins and
 * sample rate, and sets the time start index to zero.
 *
 * @param bounds Pointer to bounds2d_t whose `time.start_d`, `freq.start_d`
 *               and `freq.end_d` fields will be updated.
 * @param result STFT metadata (provides `num_frequencies` and `sample_rate`)
 *               used to map Hz values to frequency bin indices.
 */
void init_bounds(bounds2d_t *bounds, stft_t *result) {
    bounds->time.start_d = 0;

    bounds->freq.start_d = hz_to_index(result->num_frequencies, result->sample_rate, bounds->freq.start_f);
    bounds->freq.end_d   = hz_to_index(result->num_frequencies, result->sample_rate, bounds->freq.end_f);
}

/**
 * Copy a bounded rectangular block of magnitude values into a contiguous output buffer.
 *
 * Copies the frequency range [bounds->freq.start_d, bounds->freq.end_d) for each time
 * index in [bounds->time.start_d, bounds->time.end_d) from the source matrix `mags`
 * into the destination buffer `data` as contiguous rows.
 *
 * @param data Destination buffer; must be large enough to hold
 *             (bounds->time.end_d - bounds->time.start_d) * (bounds->freq.end_d - bounds->freq.start_d) floats.
 * @param mags Source magnitude matrix laid out row-major by time, with `length` floats per time row.
 * @param bounds Specifies the time and frequency start/end indices to copy (end indices are exclusive).
 * @param length Number of frequency bins (columns) in each time row of `mags`.
 */
void fast_copy(float *data, float *mags, bounds2d_t *bounds, const size_t length) {
    const size_t freq_range = bounds->freq.end_d - bounds->freq.start_d;
    const size_t copy_size = freq_range * sizeof(float);
     
    for (size_t t = bounds->time.start_d; t < bounds->time.end_d; t++) {
        memcpy(data + (t - bounds->time.start_d) * freq_range, mags + t * length + bounds->freq.start_d, copy_size);
    }
}

/**
 * Render a 2D spectrogram-like heatmap from `data` and write it to a file.
 *
 * Creates an internal heatmap of size `settings->w` x `settings->h`, maps the
 * rectangular region specified by `bounds->time` and the full vertical range
 * (height `h`) onto the heatmap, converts each sample with `brachless_db`
 * when `settings->db` is true, and accumulates points into the heatmap.
 * Frequency rows are vertically flipped so that lower frequencies appear at
 * the bottom of the image (index `h - f - 1`). The resulting heatmap is
 * saved to `settings->output_file`.
 *
 * If allocation of the heatmap fails, an error is logged and the function
 * returns without writing a file.
 *
 * @param data      Pointer to a contiguous buffer of size at least
 *                  (bounds->time.end_d - bounds->time.start_d) * settings->h,
 *                  laid out as time-major frames each containing `h` frequency
 *                  bins.
 * @param bounds    Time/frequency bounds describing the time range to render;
 *                  only `bounds->time.start_d` and `bounds->time.end_d` are
 *                  used for the horizontal extent.
 * @param settings  Plot settings (width `w`, height `h`, `db` flag, output
 *                  filename, background color and color scheme).

 **/
 
inline void plot(float *data, bounds2d_t *bounds, plot_t *settings) {
    const size_t w = settings->w;
    const size_t h = settings->h;

    heatmap_t *hm = heatmap_new(w, h);
    if (!hm) {
        ERROR("Failed to allocate heatmap");
        return;
    }

    const bool   db     = settings->db;
    const size_t tstart = bounds->time.start_d;
    const size_t tend   = bounds->time.end_d;

    #pragma omp parallel for
    for (size_t t = tstart; t < tend; t++) {
        const size_t offset = (t - tstart) * h;
        for (size_t f = 0; f < h; f++) {
            const float val = brachless_db(data[offset + f], db);
            heatmap_add_weighted_point(hm, t - tstart, h - f - 1, val);
        }
    }

    save_heatmap(&hm, settings->output_file, w, h, settings->bg_color, settings->cs_enum);
    free(hm);
}

/*
 * Apply a mel filter bank to a block of spectral frames and produce mel-band values.
 *
 * For each time frame within bounds->time.{start_d,end_d}, this function computes the
 * dot product between the frame's frequency bins (restricted to the frequency bounds)
 * and each mel filter, optionally converts the filter response to a dB-like value, and
 * stores results in a newly allocated array of size (tend - tstart) * num_filters.
 *
 * @param cont_mem   Pointer to contiguous spectral magnitude data arranged by time frames.
 *                   Only the slice defined by bounds->time and bounds->freq is read.
 * @param num_filters Number of mel filters (width of the filter bank, number of outputs per frame).
 * @param num_freq   Total number of frequency bins per filter row in mel_filter_bank (stride).
 * @param mel_filter_bank Flattened filter bank array of length num_filters * num_freq,
 *                       laid out row-major (filter index major).
 * @param bounds     Time and frequency bounds specifying the region to process.
 * @param settings   Plot/settings structure; only settings->db is used to select linear vs dB-like output.
 *
 * @return Pointer to an allocated float array of length (time_frames * num_filters) on success,
 *         or NULL on allocation failure. The caller is responsible for freeing the returned buffer.
 */
float *apply_filter_bank(float *cont_mem, size_t num_filters, size_t num_freq, float *mel_filter_bank, bounds2d_t *bounds) {
    const size_t tstart = bounds->time.start_d;
    const size_t tend   = bounds->time.end_d;
    const size_t w      = tend - tstart;
    const size_t h      = bounds->freq.end_d - bounds->freq.start_d;

    float *mel_values = (float*) malloc(w * num_filters * sizeof(float));
    if (!mel_values) {
        ERROR("Failed to allocate mel_values");
        return NULL;
    }

    /*
     #pragma omp parallel for
    for (size_t t = tstart; t < tend; t++) {
        const size_t offset = (t - tstart) * h;
        for (size_t f = 0; f < h; f++) {
            const float val = brachless_db(data[offset + f], db);
            heatmap_add_weighted_point(hm, t - tstart, h - f - 1, val);
        }
    }

    since there is 2 loops without vectorzation , this will be slower than fully vectozred dot products 
    */

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                w,              // time frames (M)
                num_filters,    // mel filters (N) 
                h,              // freq bins (K)
                1.0,            // alpha
                cont_mem, h,    // A: spectrogram [w × h]
                mel_filter_bank, num_freq, // B: filterbank [num_filters × h]
                0.0,            // beta
                mel_values, num_filters);  // C: output [w × num_filters]


    return mel_values;
}

/**
 * Compute a filtered cosine transform (FCC) over time frames.
 *
 * Applies the provided DCT-like coefficient matrix to per-frame mel-band values to
 * produce time-aligned FCC (cepstral) coefficients. Each output frame contains
 * `dft_coff->num_coff` coefficients computed as the dot product between the
 * corresponding row of `dft_coff->coeffs` and the mel values for that frame;
 * results are converted with `brachless_db` when `settings->db` is true.
 *
 * @param mel_values Pointer to input mel values laid out as contiguous frames:
 *                   frame 0 bands[0..num_filters-1], frame 1, ... . Expected length
 *                   >= (bounds->time.end_d - bounds->time.start_d) * dft_coff->num_filters.
 * @param dft_coff   Pointer to a dct_t containing `num_filters` and `num_coff`
 *                   and a coeffs array of size (num_coff * num_filters).
 * @param bounds     Time/frequency bounds; only the time range (time.start_d..time.end_d)
 *                   is used to determine the number of frames processed.
 * @param settings   Runtime settings; only `settings->db` is observed to decide
 *                   whether to convert outputs with a dB-like transform.
 *
 * @return Pointer to a newly allocated array of size (num_coff * number_of_frames)
 *         containing FCC coefficients in frame-major order, or NULL if memory
 *         allocation fails. Caller is responsible for freeing the returned buffer.
 */
float *FCC(float *mel_values, dct_t *dft_coff, bounds2d_t *bounds, plot_t *settings) {
    (void)settings;
    const size_t tstart  = bounds->time.start_d;
    const size_t tend    = bounds->time.end_d;
    const size_t w       = tend - tstart;
    const size_t num_f   = dft_coff->num_filters;
    const size_t num_c   = dft_coff->num_coff;

    float *fcc_values = (float*) malloc(w * num_c * sizeof(float));
    if (!fcc_values) {
        ERROR("Failed to allocate fcc_values");
        return NULL;
    }

    /*#pragma omp parallel for
    for (size_t t = tstart; t < tend; t++) {
        size_t offset_in = (t - tstart) * num_f;
        size_t offset_out = (t - tstart) * num_c;

        for (size_t c = 0; c < num_c; c++) {
            float sum = cblas_sdot(num_f, &dft_coff->coeffs[c * num_f], 1,
                                            &mel_values[offset_in], 1);
            fcc_values[offset_out + c] = brachless_db(sum, db);
        }
    }*/

    // SGEMM: fcc_values[w × num_c] = mel_values[w × num_f] * dft_coff->coeffs^T[num_c × num_f]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                w,        // number of rows in mel_values (time frames)
                num_c,    // number of columns in fcc_values (num_coff)
                num_f,    // inner dimension (num_f)
                1.0f,     // alpha
                &mel_values[tstart * num_f], num_f,     // A: mel_values subset
                dft_coff->coeffs, num_f,               // B: dft_coff->coeffs
                0.0f,     // beta
                fcc_values, num_c);                     // C: output fcc_values


    return fcc_values;
}


/**
 * Generate DCT-like cosine coefficients for a filter bank.
 *
 * Produces a dct_t containing a newly-allocated coefficient array of length
 * num_filters * num_coff. Coefficients are laid out with coefficient index
 * varying slower: coeffs[n * num_filters + mel].
 *
 * @param num_filters Number of mel/filter bands (width of each coefficient vector).
 * @param num_coff    Number of output coefficients per filter (number of DCT terms).
 * @return A dct_t whose `coeffs` points to a malloc'd array on success. On
 *         allocation failure `coeffs` will be NULL (an error is logged). The
 *         caller is responsible for freeing the returned `coeffs` when no
 *         longer needed.
 */
dct_t gen_cosine_coeffs(const size_t num_filters, const size_t num_coff) {
    dct_t dft_coff = {
        .num_filters = num_filters,
        .num_coff = num_coff,
        .coeffs = malloc(num_filters * num_coff * sizeof(float))
    };

    if (!dft_coff.coeffs) {
        ERROR("Failed to allocate DCT coefficients");
        return dft_coff; 
    }

    const float scale = sqrtf(2.0f / (float)num_filters);

    #pragma omp parallel for schedule(static, 256)
    for (size_t n = 0; n < num_coff; n++) {
        for (size_t mel = 0; mel < num_filters; mel++) {
            dft_coff.coeffs[n * num_filters + mel] = scale *
                cosf((float)M_PI / num_filters * (mel + 0.5f) * n);
        }
    }

    return dft_coff;
}
