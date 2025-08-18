#include "audio_tools/audio_visualizer.h"
#include "utils/bench.h"
#include "utils/cli.h"

static void finalize_settings(cli_options_t *opts, stft_t *result) {
    if (!opts || !result) return;

    // Only clamp/adjust if Mel or MFCC are enabled
    if (opts->compute_mel || opts->compute_mfcc) {

        // Compute corresponding STFT indices
        size_t start_idx = hz_to_index(result->num_frequencies, result->sample_rate, opts->min_mel_freq);
        size_t end_idx   = hz_to_index(result->num_frequencies, result->sample_rate, opts->max_mel_freq);

        // Ensure at least one bin
        if (end_idx <= start_idx) {
            LOG("Warning: max_mel_freq too low, adjusting end index from %zu to %zu", end_idx, start_idx + 1);
            end_idx = start_idx + 1;
        }

        // Clamp number of mel filters
        if (opts->num_mel_filters > end_idx - start_idx) {
            LOG("Warning: num_mel_filters (%d) > available bins (%zu), clamping to %zu",
                opts->num_mel_filters, end_idx - start_idx, end_idx - start_idx);
            opts->num_mel_filters = end_idx - start_idx;
        }

        // Clamp number of MFCC coefficients
        if (opts->compute_mfcc && opts->num_mfcc_coeffs > opts->num_mel_filters) {
            LOG("Warning: num_mfcc_coeffs (%d) > num_mel_filters (%d), clamping to %d",
                opts->num_mfcc_coeffs, opts->num_mel_filters, opts->num_mel_filters);
            opts->num_mfcc_coeffs = opts->num_mel_filters;
        }
    }
}



int main(int argc, char *argv[]) {
    cli_options_t opts;
    parse_cli(argc, argv, &opts);

    audio_data audio = auto_detect(opts.input_file);
    
    n_threads = opts.num_threads;
    opts.sr   = audio.sample_rate;

    bool valid = validate_cli_options(&opts);
   
    if(valid){

        LOG(GREEN,"✅ All inputs are valid");
        print_cli_summary(&opts);
    
    


        print_ad(&audio);

        size_t num_frequencies = opts.window_size / 2;

        // ── Precompute Window ─────────────────────────────────────────────────────
        
        START_TIMING();
        float *window_values = (float*) malloc(opts.window_size * sizeof(float));
        window_function(window_values, opts.window_size, opts.window_type);
        END_TIMING("pre:win");

        // ── FFT Plan ──────────────────────────────────────────────────────────────
        
        START_TIMING();
        fft_t fft_plan        = init_fftw_plan(opts.window_size, opts.cache_folder);
        END_TIMING("pre:fft");

        // ── Plot Settings ─────────────────────────────────────────────────────────
    
        plot_t settings       = {.cs_enum = opts.cs_stft, .db = true, .bg_color = {0,0,0,255}};

        // ── STFT ──────────────────────────────────────────────────────────────────
        
        START_TIMING();
        stft_t result         = stft(&audio, opts.window_size, opts.hop_size, window_values, &fft_plan);
        END_TIMING("stft");


        finalize_settings(&opts,&result);

        print_stft_bench(&result.benchmark);

        bounds2d_t bounds     = {0};
        bounds.freq.start_f   = opts.min_mel_freq;
        bounds.freq.end_f     = opts.max_mel_freq;  // No need for fallback check, finalize_audio_dependent_defaults() handles this

        init_bounds(&bounds, &result);
        set_limits(&bounds, result.num_frequencies, result.output_size);

        const size_t t_len   = bounds.time.end_d - bounds.time.start_d;
        const size_t f_len   = bounds.freq.end_d - bounds.freq.start_d;
        float *cont_mem      = malloc(t_len * f_len * sizeof(float));

        START_TIMING();
        fast_copy(cont_mem, result.magnitudes, &bounds, result.num_frequencies);
        END_TIMING("copy");

        sprintf(settings.output_file, "%s_stft.png", opts.output_base);
        settings.w           = t_len;
        settings.h           = f_len;
        
        START_TIMING();
        plot(cont_mem, &bounds, &settings);
        END_TIMING("plt:stft");

        float       *mel_values = NULL;
        float       *fcc_values = NULL;
        float       *filterbank = NULL;
        filter_bank_t bank;
        dct_t         dft_coff;

        // ── Filterbank Processing ────────────────────────────────────────────────
        
        if (opts.compute_mel || opts.compute_mfcc) {
            START_TIMING();
            filterbank         = (float*) calloc((num_frequencies + 1) * (opts.num_mel_filters + 2), sizeof(float));
            LOG("Processing filterbank type: %d", opts.filterbank_type);
            bank               = gen_filterbank(opts.filterbank_type, opts.min_mel_freq, opts.max_mel_freq,
                                                opts.num_mel_filters, audio.sample_rate, opts.window_size, filterbank);
            END_TIMING("pre:mel");
        }

        // ── Mel Spectrogram ───────────────────────────────────────────────────────
        
        if (opts.compute_mel) {
            START_TIMING();
            mel_values         = apply_filter_bank(cont_mem, opts.num_mel_filters, result.num_frequencies, filterbank, &bounds, &settings);
            END_TIMING("mel");

            sprintf(settings.output_file, "%s_mel.png", opts.output_base);
            settings.h         = opts.num_mel_filters;
            settings.w         = t_len;
            
            START_TIMING();
            plot(mel_values, &bounds, &settings);
            END_TIMING("plt:mel");
        }

        // ── MFCC Processing ───────────────────────────────────────────────────────
        
        if (opts.compute_mfcc) {
            START_TIMING();
            dft_coff           = gen_cosine_coeffs(opts.num_mel_filters, opts.num_mfcc_coeffs);
            fcc_values         = FCC(mel_values, &dft_coff, &bounds, &settings);
            END_TIMING("mfcc");

            sprintf(settings.output_file, "%s_mfcc.png", opts.output_base);
            settings.h         = dft_coff.num_coff;
            settings.w         = t_len;
            
            START_TIMING();
            plot(fcc_values, &bounds, &settings);
            END_TIMING("plt:mfcc");
        }

        // ── Cleanup ───────────────────────────────────────────────────────────────
        
        free_stft(&result);
        free_audio(&audio);
        free(window_values);
        free_fft_plan(&fft_plan);
        free(cont_mem);
        
        if (mel_values) free(mel_values);
        if (fcc_values) free(fcc_values);
        if (filterbank) {
            free(filterbank);
            free(bank.freq_indexs);
            free(bank.weights);
        }
        if (dft_coff.coeffs) free(dft_coff.coeffs);

        print_bench_ranked();

    }
    else{
        ERROR("Input Validation failed");
    }

    return 0;
}