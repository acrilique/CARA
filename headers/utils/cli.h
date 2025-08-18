#ifndef CLI_H
#define CLI_H

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <stdbool.h>
    #include "audio_tools/spectral_features.h"  

    #define DEFAULT_WINDOW_SIZE       1024
    #define DEFAULT_HOP_SIZE          512
    #define DEFAULT_WINDOW_TYPE       "hanning"
    #define DEFAULT_NUM_MEL_FILTERS   40
    #define DEFAULT_NUM_MFCC_COEFFS   13
    #define DEFAULT_COLOR_SCHEME      0
    #define DEFAULT_FILTERBANK_TYPE   F_MEL
    #define DEFAULT_CACHE_FOLDER      "cache"

    typedef struct {
        const char       *input_file;
        const char       *output_base;
        int               window_size;
        int               hop_size;
        const char       *window_type;
        size_t            num_mel_filters;
        float             min_mel_freq;
        float             max_mel_freq;
        unsigned short    num_mfcc_coeffs;
        bool              compute_mel;
        bool              compute_mfcc;
        unsigned short    cs_stft;
        unsigned short    cs_mel;
        unsigned short    cs_mfcc;
        int               filterbank_type;
        const char       *cache_folder;
        int               num_threads;
        float             sr;
    } cli_options_t;

    void parse_cli(int argc, char *argv[], cli_options_t *opts);
    bool validate_cli_options(cli_options_t *opts);
    void print_cli_summary(const cli_options_t *opts);

#endif // CLI_H
