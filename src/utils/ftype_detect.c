#include "utils/ftype_detect.h"
#include "utils/bench.h"

const char* mime_types_map[AUDIO_TYPE_COUNT] = {
    "audio/unknown",  // AUDIO_UNKNOWN
    "audio/wav",      // AUDIO_WAV
    "audio/mpeg",     // AUDIO_MPEG 
    "audio/flac",     // AUDIO_FLAC
    "audio/opus",     // AUDIO_OPUS 
    "audio/ogg",      // AUDIO_OGG
    "audio/aac",      // AUDIO_AAC
    "audio/amr"       // AUDIO_AMR
};

/**
 * Detect the audio type of a file by inspecting its header and set its size.
 *
 * Reads up to MAX_HEADER_SIZE bytes from the file named by `filename`, examines
 * common audio container and codec signatures (WAV/RIFF, MP3/ID3 or MPEG frame,
 * FLAC, OGG/Opus, AAC ADTS, AMR) and returns the corresponding audio_type.
 *
 * `file_size` is written with the file size in bytes; it must point to a valid
 * long and is required by the function. The function opens the file in binary
 * mode and closes it before returning.
 *
 * On error (NULL filename, I/O failure, insufficient header bytes) or when no
 * signature matches, AUDIO_UNKNOWN is returned and *file_size is not modified.
 *
 * @param filename Path to the file to inspect (must be non-NULL).
 * @param file_size Output pointer to receive the file size in bytes (must be non-NULL).
 * @return Detected audio_type enum value (AUDIO_UNKNOWN if unknown or on error).
 */
audio_type detect_audio_type(const char *filename, long *file_size) {
    if (!filename) {
        return AUDIO_UNKNOWN;
    }

    FILE *file = fopen(filename, "rb");
    if (!file) {
        ERROR("Error opening file");
        return AUDIO_UNKNOWN;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        ERROR("Error seeking file end");
        fclose(file);
        return AUDIO_UNKNOWN;
    }

    *file_size = ftell(file);
    if (*file_size < 0) {
        ERROR("Error getting file size");
        fclose(file);
        return AUDIO_UNKNOWN;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        ERROR("Error seeking file beginning");
        fclose(file);
        return AUDIO_UNKNOWN;
    }

    uint8_t buffer[MAX_HEADER_SIZE];
    size_t read_bytes = fread(buffer, 1, MAX_HEADER_SIZE, file);
    fclose(file);

    if (read_bytes < 12) {
        return AUDIO_UNKNOWN;
    }

    uint32_t riff_sig, wave_sig, flac_sig, ogg_sig, amr_sig;
    
    memcpy(&riff_sig, buffer, 4);
    memcpy(&wave_sig, buffer + 8, 4);
    memcpy(&flac_sig, buffer, 4);
    memcpy(&ogg_sig, buffer, 4);
    memcpy(&amr_sig, buffer, 4);

    // Check for WAV: "RIFF" + "WAVE"
    const int is_wav  = (riff_sig == 0x46464952) && (wave_sig == 0x45564157);
    // Check for MP3: "ID3" or MPEG frame
    const int is_mp3  = ((buffer[0] == 0x49 && buffer[1] == 0x44 && buffer[2] == 0x33) || ((buffer[0] & 0xFF) == 0xFF && (buffer[1] & 0xE0) == 0xE0));
    // Check for FLAC: "fLaC"
    const int is_flac = (flac_sig == 0x43614C66);
    // Check for OGG: "OggS"
    const int is_ogg  = (ogg_sig == 0x5367674F);
    // Check for OPUS: "OggS" followed by "OpusHead"
    const int is_opus = is_ogg && (memcmp(buffer + 28, "OpusHead", 8) == 0);
    // Check for AAC: ADTS Sync word
    const int is_aac  = ((buffer[0] == 0xFF) && ((buffer[1] & 0xF0) == 0xF0));
    // Check for AMR: "#!AMR"
    const int is_amr  = (amr_sig == 0x524D4123);

    audio_type type = AUDIO_UNKNOWN;

    type = is_wav  * AUDIO_WAV  |
           is_mp3  * AUDIO_MPEG |
           is_flac * AUDIO_FLAC |
           is_opus * AUDIO_OPUS |
           is_ogg  * AUDIO_OGG  |
           is_aac  * AUDIO_AAC  |
           is_amr  * AUDIO_AMR;

    LOG("%s auto detected to be %s", filename, mime_types_map[type]);

    return type;
}

/**
 * Return the MIME type string for a given audio_type.
 *
 * If `type` is within the valid range [0, AUDIO_TYPE_COUNT), the corresponding
 * MIME string from the internal mapping is returned. For any out-of-range
 * value the function returns the MIME string for unknown audio ("audio/unknown").
 *
 * @param type Audio type enum value to query.
 * @return Pointer to a NUL-terminated MIME type string (owned by the caller's
 *         process; do not free).
 */
inline const char* get_mime_type(audio_type type) {
    if (type >= 0 && type < AUDIO_TYPE_COUNT) {
        return mime_types_map[type];
    }
    return mime_types_map[0]; 
}