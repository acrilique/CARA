#include "audio_tools/audio_io.h"
#include "audio_tools/minimp3.h"
#include "utils/bench.h"

/**
 * Print summary metadata for an audio_data object and return its duration in seconds.
 *
 * If `data` is NULL or has no samples, this function logs an error and returns 0.0.
 * Otherwise it computes duration as num_samples / (sample_rate * channels),
 * logs duration, channel count, sample count, sample rate, and file size, and
 * returns the computed duration.
 *
 * @param data Pointer to a populated audio_data structure. Must have valid
 *             `.samples`, `.num_samples`, `.sample_rate`, and `.channels` for
 *             meaningful results.
 * @return Duration of the audio in seconds, or 0.0 on invalid input.
 */


float print_ad(audio_data *data) {
    if (!data || !data->samples) {
        ERROR("Invalid audio data");
        return 0.0f;
    }
    
    const float duration = (float)data->num_samples / ((float)(data->sample_rate * data->channels));
    
    LOG("duration:%.3f", duration);
    LOG("channels:%zu", data->channels);
    LOG("num_samples:%zu", data->num_samples);
    LOG("sample_rate:%.4f", data->sample_rate);
    LOG("file_size_bytes:%ld", data->file_size);

    return duration;
}

void free_audio(audio_data *audio) {
    if (audio) {
        if (audio->samples) { 
            free(audio->samples);
            audio->samples = NULL;
        }
    }
}

/**
 * Read an entire file into memory.
 *
 * Returns a file_buffer whose `data` points to a newly allocated buffer containing
 * the file contents and whose `size` is the number of bytes read. On any error
 * (open, fstat, allocation, or read failure) returns a file_buffer with `data`
 * == NULL and `size` == 0.
 *
 * The caller is responsible for freeing `data` when no longer needed.
 *
 * @return file_buffer populated with file data and size, or {NULL, 0} on error.
 */
file_buffer read_file(const char *filename) {
    file_buffer result = {.data = NULL, .size = 0};

    FILE *fin = fopen(filename, "rb");
    if (!fin) {
        ERROR("Failed to open file '%s'", filename);
        return result;
    }

    struct stat st;
    if (fstat(fileno(fin), &st) != 0) {
        ERROR("fstat failed for file '%s'", filename);
        fclose(fin);
        return result;
    }

    result.size = (uint64_t)st.st_size;
    result.data = (uint8_t *)malloc(result.size);
    if (!result.data) {
        ERROR("Memory allocation failed for file '%s' (%lu bytes)", filename, result.size);
        fclose(fin);
        return result;
    }

    if (fread(result.data, 1, result.size, fin) != result.size) {
        ERROR("fread failed for file '%s'", filename);
        free(result.data);
        result.data = NULL;
        result.size = 0;
    }

    fclose(fin);
    return result;
}

/**
 * Read a WAV file and decode it into an audio_data structure.
 *
 * Opens the WAV file via libsndfile, allocates a float buffer for PCM samples,
 * and fills audio.num_samples, audio.samples, audio.sample_rate, and audio.channels.
 * If the full number of frames cannot be read, num_samples is adjusted to the
 * number of frames actually read. The provided file_size is stored in
 * audio.file_size but is not used for decoding.
 *
 * @param filename Path to the WAV file to read.
 * @param file_size Size of the file in bytes; stored in the returned audio_data.
 * @return An audio_data value. On success, `samples` points to a malloc'd
 *         float buffer of interleaved PCM samples and `num_samples` reflects
 *         the total sample count (frames * channels). On failure, `samples`
 *         is NULL and `num_samples` is zero.
 */
audio_data read_wav(const char *filename, long file_size) {
    audio_data audio = {0};
    audio.file_size = file_size;

    SNDFILE *file;
    SF_INFO sf_info = {0};  

    file = sf_open(filename, SFM_READ, &sf_info);
    if (!file) {
        ERROR("Error opening file");
        return audio;
    }

    audio.num_samples = (size_t)sf_info.frames * sf_info.channels;
    audio.samples = (float*)malloc(audio.num_samples * sizeof(float));

    if (!audio.samples) {
        ERROR("Memory allocation failed");
        sf_close(file);
        return audio;
    }

    sf_count_t frames_read = sf_readf_float(file, audio.samples, sf_info.frames);
    if (frames_read < sf_info.frames) {
        WARN("Error reading audio data: read %lld of %lld frames", 
                (long long)frames_read, (long long)sf_info.frames);
        audio.num_samples = (size_t)frames_read * sf_info.channels;
    }

    audio.sample_rate = sf_info.samplerate;
    audio.channels = sf_info.channels;

    sf_close(file);
    return audio;
}


/**
 * Scan an in-memory MP3 file buffer and record the byte offsets of successive MP3 frames.
 *
 * This function walks the provided buffer, locating MP3 frames using mp3d_find_frame and
 * recording the advance (bytes consumed for each frame) into a newly allocated array.
 * The returned frames structure contains the array of per-frame byte advances, the number
 * of frames found, and the average bytes per frame (0.0 if no frames found).
 *
 * The function allocates memory for the returned frames.data; the caller is responsible
 * for freeing that buffer when no longer needed. On allocation failure the returned
 * frames.data will be NULL and frames.count will be 0.
 *
 * @param buf Pointer to a file_buffer containing MP3 data; must be non-NULL and have a valid size.
 * @return A frames struct with:
 *   - data: dynamically allocated array of unsigned short entries (per-frame byte advances) or NULL on allocation failure,
 *   - count: number of frames discovered,
 *   - avg_byte_per_frame: average bytes per frame (0.0 if count is 0).
 */
frames find_mp3_frame_offsets(file_buffer *buf) {
    frames result = {.data = NULL, .count = 0, .avg_byte_per_frame = 0.0f};

    uint64_t offset       = 0;
    int free_format_bytes = 0;
    int frame_bytes       = 0;

    const int max_frames = buf->size / WORST_CASE_FRAME_SIZE;   // see audio_io.h line 5

    result.data = (unsigned short *)malloc(max_frames * sizeof(unsigned short));
    if (!result.data) {
        ERROR("Memory allocation failed for %d frames", max_frames);
        return result;
    }

    int frame_index = 0;
    int total_bytes = 0;

   while (offset < buf->size && frame_index < max_frames) {
        int next_offset = mp3d_find_frame(buf->data + offset, buf->size - offset, &free_format_bytes, &frame_bytes);

        if (frame_bytes == 0 || next_offset < 0)
            break;

        const int total_advance = next_offset + frame_bytes;

        result.data[frame_index++] = (unsigned short)total_advance;
        total_bytes += total_advance;
        offset += total_advance;
    }

    result.count = frame_index;
    if (frame_index > 0)
        result.avg_byte_per_frame = (float)total_bytes / frame_index;

    return result;
}


/**
 * Decode an MP3 file into PCM samples and return as an audio_data structure.
 *
 * Reads the file at `filename`, decodes MP3 frames using the bundled minimp3
 * decoder and allocates a heap buffer for interleaved PCM samples. The
 * returned audio_data contains populated fields: `samples` (heap-allocated),
 * `num_samples`, `channels`, `sample_rate`, and `file_size` (set from the
 * `file_size` parameter). On failure the returned audio_data will be zeroed
 * and `samples` will be NULL.
 *
 * @param filename Path to the MP3 file to decode.
 * @param file_size Size of the input file (stored in the returned audio_data);
 *        the function will still read and decode the file from disk regardless
 *        of this value.
 * @return audio_data with decoded PCM samples (caller is responsible for
 *         freeing `samples`, e.g. with free_audio). If decoding or allocation
 *         fails, returns an audio_data with `samples == NULL` and `num_samples == 0`.
 */
audio_data read_mp3(const char *filename, long file_size) {
    audio_data audio = {0};
    audio.file_size = file_size;

    static mp3dec_t mp3d;
    mp3dec_init(&mp3d);

    START_TIMING();
    file_buffer buf = read_file(filename);
    END_TIMING("file_read");

    if (!buf.data || buf.size == 0) {
        ERROR("Failed to read input file: %s", filename);
        return audio;
    }

    START_TIMING();
    frames f = find_mp3_frame_offsets(&buf);
    END_TIMING("frame_scan");

    LOG("Total frames: %d", f.count);
    LOG("Average frame size: %.2f bytes", f.avg_byte_per_frame);

    // Estimate max PCM samples (safe upper bound at 32kbps)
    const size_t max_pcm_samples = (buf.size * MINIMP3_MAX_SAMPLES_PER_FRAME) / 32;
    const size_t pcm_bsiz        = max_pcm_samples * sizeof(PARAM_DATATYPE) * 2;

    audio.samples = malloc(pcm_bsiz);
    if (!audio.samples) {
        ERROR("Memory allocation failed");
        free(buf.data);
        free(f.data);
        return audio;
    }

    PARAM_DATATYPE *full_pcm = (PARAM_DATATYPE *)audio.samples;
    uint64_t decoded_samples = 0;

    uint8_t *input_ptr     = buf.data;
    int      remaining_size = buf.size;

    START_TIMING();
    while (remaining_size > 0) {
        mp3dec_frame_info_t info;
        PARAM_DATATYPE pcm[MINIMP3_MAX_SAMPLES_PER_FRAME * 2];

        int samples = mp3dec_decode_frame(&mp3d, input_ptr, remaining_size, pcm, &info);

        if (info.frame_bytes == 0 || remaining_size < info.frame_bytes)
            break;

        if (samples > 0) {
            if ((decoded_samples + samples) * info.channels > max_pcm_samples) {
                ERROR("PCM buffer overflow prevented");
                break;
            }

            const size_t copy_size = samples * sizeof(PARAM_DATATYPE) * info.channels;
            memcpy(full_pcm + (decoded_samples * info.channels), pcm, copy_size);
            decoded_samples += samples;

            audio.channels = info.channels;
            audio.sample_rate = info.hz;
        }

        input_ptr += info.frame_bytes;
        remaining_size -= info.frame_bytes;
    }
    END_TIMING("dec_mp3");

    audio.num_samples = decoded_samples * audio.channels;

    free(buf.data);
    free(f.data);

    return audio;
}


/**
 * Auto-detect audio file format and decode into PCM audio_data.
 *
 * Detects the file type for the given filename and dispatches to the appropriate
 * decoder (WAV or MP3). On success returns an audio_data populated with sample
 * buffer, sample count, sample rate, channels, and file_size. If the format is
 * unsupported or decoding fails, an empty/zeroed audio_data is returned and an
 * error is logged.
 *
 * @param filename Path to the input audio file to inspect and decode.
 * @return Decoded audio_data structure. Fields will be zero/NULL on failure.
 */
audio_data auto_detect(const char *filename) {
    audio_data audio = {0};
    
    START_TIMING();
    audio_type type = detect_audio_type(filename, &audio.file_size);
    END_TIMING("auto_det");

    switch (type) {
        case AUDIO_WAV:
            audio = read_wav(filename, audio.file_size);
            END_TIMING("dec_wav");
            break;
        case AUDIO_MPEG:
            audio = read_mp3(filename, audio.file_size);
            break;
        default:
            ERROR("Unknown or unsupported audio format");
            break;
    }
    
    return audio;
}