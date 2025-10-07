#include "png_tools/png_tools.h"
#include "utils/bench.h"

/**
 * Apply a background color to an RGBA image in-place.
 *
 * For each pixel, if the RGB components are all zero (pure black) they are
 * replaced with the provided color's RGB components. The alpha component of
 * every pixel is set to the provided color's alpha.
 *
 * The function operates in-place on a tightly packed 4-byte-per-pixel RGBA
 * buffer and does not allocate memory.
 *
 * @param image Pointer to the image buffer (must contain at least width * height * 4 bytes).
 *              Pixels are laid out as interleaved RGBA bytes.
 * @param width Width of the image in pixels.
 * @param height Height of the image in pixels.
 * @param color Four-byte RGBA color (color[0]=R, color[1]=G, color[2]=B, color[3]=A).
 */
void add_bg(unsigned char *image, size_t width, size_t height, unsigned char color[4]) {
    const size_t num_pixels = width * height;
    
    for (size_t i = 0; i < num_pixels; i++) {
        unsigned char *pixel = &image[i * 4];
        if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
            pixel[0] = color[0]; // Red
            pixel[1] = color[1]; // Green
            pixel[2] = color[2]; // Blue
        }
        pixel[3] = color[3];
    }
}

/**
 * Write an 8-bit RGBA image buffer to a PNG file.
 *
 * The provided image buffer is written as an 8-bit per channel RGBA PNG (no interlace).
 * The function allocates a temporary row buffer and writes the image row-by-row using libpng.
 * On failure (file open, libpng initialization, or libpng error), the function logs an error
 * and terminates the process via exit(1).
 *
 * @param filename Path to the output PNG file to create/overwrite.
 * @param image Pointer to the source image pixels in row-major order (RGBA, 4 bytes per pixel).
 * @param width Image width in pixels.
 * @param height Image height in pixels.
 */
void save_png(const char *filename, const unsigned char *image, size_t width, size_t height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        ERROR("Failed to open file for writing");
        exit(1);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        ERROR("png_create_write_struct failed");
        fclose(fp);
        exit(1);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        ERROR("png_create_info_struct failed");
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        exit(1);
    }

    if (setjmp(png_jmpbuf(png))) {
        ERROR("Error during PNG creation");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        exit(1);
    }

    png_init_io(png, fp);
    png_set_IHDR(
        png,
        info,
        (png_uint_32)width, (png_uint_32)height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);

    png_bytep row = (png_bytep)malloc(4 * width);
    for (size_t y = 0; y < height; ++y) {
        memcpy(row, image + 4 * width * y, 4 * width);
        png_write_row(png, row);
    }

    LOG("%s saved", filename);

    free(row);

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

/**
 * Resize an RGBA image using nearest-neighbor sampling.
 *
 * The input image is expected as a contiguous row-major buffer with 4 bytes per pixel
 * (R,G,B,A). The function allocates and returns a new buffer of size new_width * new_height * 4.
 * The caller is responsible for freeing the returned buffer.
 *
 * @param original Pointer to the source image buffer (RGBA, row-major).
 * @param orig_width Width of the source image in pixels.
 * @param orig_height Height of the source image in pixels.
 * @param new_width Desired width of the resized image in pixels.
 * @param new_height Desired height of the resized image in pixels.
 * @return Pointer to a newly allocated RGBA buffer containing the resized image.
 */
unsigned char* resize_image(const unsigned char *original, size_t orig_width, size_t orig_height, size_t new_width, size_t new_height) {

    unsigned char *resized = (unsigned char *)malloc(new_width * new_height * 4);
    if (!resized) {
        ERROR("Memory allocation failed for image resize");
        exit(1);
    }

    const float  x_scale = (float)orig_width / new_width;
    const float  y_scale = (float)orig_height / new_height;

    const unsigned char *orig_row_ptr;
    unsigned char       *resized_row_ptr = resized;

    const size_t orig_row_size    = orig_width * 4;
    const size_t resized_row_size = new_width * 4;

    for (size_t y = 0; y < new_height; ++y) {
        orig_row_ptr = original + (size_t)(y * y_scale) * orig_row_size;
        unsigned char *resized_pixel_ptr = resized_row_ptr;

        for (size_t x = 0; x < new_width; ++x) {
            const size_t orig_x = (size_t)(x * x_scale) * 4;
            memcpy(resized_pixel_ptr, orig_row_ptr + orig_x, 4); 
            resized_pixel_ptr += 4;
        }
        resized_row_ptr += resized_row_size;
    }
    
    LOG("Image resized from (%zu x %zu) -> (%zu x %zu)", orig_width, orig_height, new_width, new_height);

    return resized;
}
