#include "utils/cli.h"
#include "utils/bench.h"
#include "libheatmap/heatmap_tools.h"


/**
 * Parse command-line arguments and populate a cli_options_t with the selected settings.
 *
 * Initializes opts with default values and updates fields according to recognized
 * long and short options (input/output paths, STFT window/hop/window type,
 * MEL/MFCC parameters, color schemes, filterbank type, cache folder, and thread count).
 * If an unknown filter bank name is provided, the function logs an error and terminates
 * the process via exit(1).
 *
 * @param argc Number of command-line arguments.
 * @param argv Argument vector as passed to main().
 * @param opts Pointer to a preallocated cli_options_t that will be initialized and filled.
 */
void parse_cli(int argc, char *argv[], cli_options_t *opts) {
    


    opts->window_size       = DEFAULT_WINDOW_SIZE;
    opts->hop_size          = DEFAULT_HOP_SIZE;
    opts->window_type       = DEFAULT_WINDOW_TYPE;
    opts->num_mel_filters   = DEFAULT_NUM_MEL_FILTERS;
    opts->min_mel_freq      = 0;
    opts->max_mel_freq      = 0;
    opts->num_mfcc_coeffs   = DEFAULT_NUM_MFCC_COEFFS;
    opts->compute_mel       = false;
    opts->compute_mfcc      = false;
    opts->cs_stft           = 999;
    opts->cs_mel            = 999;
    opts->cs_mfcc           = 999;
    opts->filterbank_type   = DEFAULT_FILTERBANK_TYPE;
    opts->cache_folder      = DEFAULT_CACHE_FOLDER;
    opts->num_threads       = n_threads;
    opts->input_file        = NULL;
    opts->output_base       = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--input_audio_file") || !strcmp(argv[i], "-i")) {
            opts->input_file  = argv[++i];
        } else if (!strcmp(argv[i], "--output_base") || !strcmp(argv[i], "-o")) {
            opts->output_base = argv[++i];
        } else if (!strcmp(argv[i], "--fft_window_size") || !strcmp(argv[i], "-ws")) {
            opts->window_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--hop_length") || !strcmp(argv[i], "-hop")) {
            opts->hop_size    = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--window_function") || !strcmp(argv[i], "-wf")) {
            opts->window_type = argv[++i];
        } else if (!strcmp(argv[i], "--num_mel_filters") || !strcmp(argv[i], "-nm")) {
            opts->compute_mel =  true;
            opts->num_mel_filters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--min_mel_freq") || !strcmp(argv[i], "-mi_feq")) {
            opts->min_mel_freq    = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--max_mel_freq") || !strcmp(argv[i], "-mx_feq")) {
            opts->max_mel_freq    = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--num_mfcc_coeffs") || !strcmp(argv[i], "-nfcc")) {
            opts->compute_mfcc    = true;
            opts->num_mfcc_coeffs = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--stft_color_scheme") || !strcmp(argv[i], "-stft_cs")) {
            opts->cs_stft = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--mel_color_scheme") || !strcmp(argv[i], "-fb_cs")) {
            opts->cs_mel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fcc_color_scheme") || !strcmp(argv[i], "-fcc_cs")) {
            opts->cs_mfcc = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--filter_bank_type") || !strcmp(argv[i], "-fb")) {
            const char *val = argv[++i];
            if (!strcmp(val, "mel"))        opts->filterbank_type = F_MEL;
            else if (!strcmp(val, "bark"))  opts->filterbank_type = F_BARK;
            else if (!strcmp(val, "erb"))   opts->filterbank_type = F_ERB;
            else if (!strcmp(val, "chirp")) opts->filterbank_type = F_CHIRP;
            else if (!strcmp(val, "cam"))   opts->filterbank_type = F_CAM;
            else if (!strcmp(val, "log10")) opts->filterbank_type = F_LOG10;
            else {
                ERROR("Unknown filterbank type: %s", val);
                exit(1);
            }

            opts->compute_mel  = true;
            opts->compute_mfcc = true;
        } else if (!strcmp(argv[i], "--cache_folder") || !strcmp(argv[i], "-c")) {
            opts->cache_folder = argv[++i];
        } else if (!strcmp(argv[i], "--num_threads") || !strcmp(argv[i], "-t")) {
            opts->num_threads = atoi(argv[++i]);
        }
    }

}
/**
 * Print a formatted summary of the resolved command-line options.
 *
 * Outputs STFT settings (always), and MEL and MFCC sections only when enabled in
 * the provided options. Also prints thread count, cache folder, input file, and
 * output base. Color-scheme enum values are resolved to human-readable names
 * when available. If `opts` is NULL the function does nothing.
 *
 * @param opts Pointer to the populated cli_options_t describing current options.
 */
void print_cli_summary(const cli_options_t *opts) {
    if (!opts) return;

    // Header
    LOG("%s════════════════════════════════════════════════════%s", BRIGHT_CYAN, RESET);
    LOG("%s           CLI OPTIONS SUMMARY                     %s", BRIGHT_CYAN, RESET);
    LOG("%s════════════════════════════════════════════════════%s", BRIGHT_CYAN, RESET);

    // STFT settings
    char *stft_str = (opts->cs_stft != 999) ? cs_from_enum(opts->cs_stft, false) : NULL;
    LOG("%sSTFT  :%s window_size=%d, hop_size=%d, window_type=%s, color_scheme=%s",
        BRIGHT_BLUE, RESET,
        opts->window_size,
        opts->hop_size,
        opts->window_type ? opts->window_type : "default",
        stft_str ? stft_str : "default");
    if (stft_str) free(stft_str);

    // MEL settings
    if (opts->compute_mel) {
        char *mel_str = (opts->cs_mel != 999) ? cs_from_enum(opts->cs_mel, false) : NULL;
        LOG("%sMEL   :%s num_filters=%zu, min_freq=%.1f Hz, max_freq=%.1f Hz, filterbank=%d, color_scheme=%s",
            BRIGHT_MAGENTA, RESET,
            opts->num_mel_filters,
            opts->min_mel_freq,
            opts->max_mel_freq,
            opts->filterbank_type,
            mel_str ? mel_str : "default");
        if (mel_str) free(mel_str);
    }

    // MFCC settings
    if (opts->compute_mfcc) {
        char *mfcc_str = (opts->cs_mfcc != 999) ? cs_from_enum(opts->cs_mfcc, false) : NULL;
        LOG("%sMFCC  :%s num_coeffs=%zu, color_scheme=%s",
            BRIGHT_YELLOW, RESET,
            opts->num_mfcc_coeffs,
            mfcc_str ? mfcc_str : "default");
        if (mfcc_str) free(mfcc_str);
    }

    // Misc settings
    LOG("%sThreads:%s %d", BRIGHT_GREEN, RESET, opts->num_threads);
    LOG("%sCache  :%s %s", BRIGHT_GREEN, RESET, opts->cache_folder ? opts->cache_folder : "(none)");
    LOG("%sInput  :%s %s", BRIGHT_GREEN, RESET, opts->input_file  ? opts->input_file  : "(none)");
    LOG("%sOutput :%s %s", BRIGHT_GREEN, RESET, opts->output_base ? opts->output_base : "(none)");

    // Footer
    LOG("%s════════════════════════════════════════════════════%s", BRIGHT_CYAN, RESET);
}




/**
 * Validate and sanitize parsed CLI options.
 *
 * Performs consistency checks on the provided cli_options_t, ensures required
 * fields are present, clamps out-of-range values, and applies safe defaults.
 * Side effects: may modify fields inside `opts` (e.g., clamp `num_threads` to
 * [1, omp_get_max_threads()], set `window_type` to "hann" if unknown, clamp
 * `min_mel_freq` to 0, clamp `max_mel_freq` to the Nyquist frequency, and
 * adjust other fields as needed).
 *
 * @param opts Pointer to the options structure to validate and sanitize.
 *             Must not be NULL.
 * @return true if the options passed all validation checks; false if `opts` is
 *         NULL or if required/semantic validation failed (for example missing
 *         required paths, invalid mel/mfcc configuration, or invalid filter
 *         bank type).
 */

bool validate_cli_options(cli_options_t *opts) {
    const unsigned int mx_thrds = omp_get_max_threads();

    if (!opts) {
        ERROR("CLI options structure is NULL");
        return false;
    }

    bool valid = true;

    // ─── Required Arguments ───────────────────────────────────────
    if (!opts->input_file || !opts->output_base) {
        ERROR("Missing required arguments: --input_audio_file and --output_base");
        valid = false;
    }

    // ─── Thread Count ─────────────────────────────────────────────
    if (opts->num_threads < 1) {
        WARN("Invalid thread count %d, defaulting to 1", opts->num_threads);
        opts->num_threads = 1;
    }
   if ((unsigned int)opts->num_threads > mx_thrds) {
        WARN("Requested threads %d exceeds max %d, clamping", 
             opts->num_threads, mx_thrds);
        opts->num_threads = mx_thrds;
    }

    // ─── Mel / MFCC Validation ────────────────────────────────────
    if (opts->compute_mel || opts->compute_mfcc) {
        if (opts->num_mel_filters < 1) {
            ERROR("Number of mel filters %zu must be >= 1", opts->num_mel_filters);
            valid = false;
        }
        if (opts->compute_mfcc && opts->num_mfcc_coeffs > opts->num_mel_filters) {
            ERROR("MFCC coeffs %zu cannot exceed mel filters %zu",opts->num_mfcc_coeffs, opts->num_mel_filters);
            valid = false;
        }
        if (opts->min_mel_freq < 0) {
            ERROR("Minimum mel frequency %.1f must be non-negative", opts->min_mel_freq);
            valid = false;
        }
        if (opts->max_mel_freq > 0 && opts->max_mel_freq <= opts->min_mel_freq) {
            ERROR("Max mel frequency %.1f must be greater than min %.1f", opts->max_mel_freq, opts->min_mel_freq);
            valid = false;
        }

        // ─── Filterbank Type ──────────────────────────────────────
        switch (opts->filterbank_type) {
            case F_MEL:
            case F_BARK:
            case F_ERB:
            case F_CHIRP:
            case F_CAM:
            case F_LOG10:
                break;
            default:
                ERROR("Invalid filterbank type: %d", opts->filterbank_type);
                valid = false;
        }
    }

    // ─── Window Type ──────────────────────────────────────────────
    const char *valid_windows[] = {
        "hann","hamming","blackman",
        "blackman-harris","bartlett",
        "flattop","gaussian","kaiser",
        NULL
    };
    bool win_ok = false;
    for (int i = 0; valid_windows[i]; i++) {
        if (!strcmp(opts->window_type, valid_windows[i])) {
            win_ok = true;
            break;
        }
    }
    if (!win_ok) {
        WARN("Unknown window type '%s', defaulting to 'hann'", opts->window_type);
        opts->window_type = "hann";
    }

    if (opts->min_mel_freq < 0) {
        WARN("Warning: min_mel_freq (%f) < 0, clamping to 0", opts->min_mel_freq);
        opts->min_mel_freq = 0;
    }

    size_t nyquist = opts->sr / 2;
    if (opts->max_mel_freq > nyquist || opts->max_mel_freq < 1) {
        WARN("Warning: max_mel_freq (%f) > Nyquist (%zu), clamping to %zu",opts->max_mel_freq, nyquist, nyquist);
        opts->max_mel_freq = nyquist;
    }


    return valid;
}
