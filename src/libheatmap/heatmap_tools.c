#include "libheatmap/heatmap_tools.h"
#include "utils/bench.h"
/**
 * Save a heatmap to a PNG file using a specified color scheme and background color.
 *
 * Renders the provided heatmap into an RGBA image, applies the given background color,
 * and writes the result to the specified PNG file. This function frees the heatmap
 * and the loaded color scheme before returning.
 *
 * @param hm Pointer to the heatmap pointer; the pointed heatmap will be freed on success.
 * @param output_file Path to the output PNG file to write.
 * @param w Image width in pixels.
 * @param h Image height in pixels.
 * @param bg_clr RGBA background color as an array of 4 unsigned chars (order: R,G,B,A).
 * @param cs_enum Index of the color scheme to use (must be in range 0..NUM_CS_MAX-1).
 * @return 0 on success; -1 if cs_enum is out of range.
 */
int save_heatmap(heatmap_t **hm,char *output_file,size_t w,size_t h,unsigned char bg_clr[4],int cs_enum){
    
    if(cs_enum > NUM_CS_MAX-1) return -1;

        
    unsigned char *image = malloc(sizeof(unsigned char) * w * h * 4);
    
    heatmap_colorscheme_t *scheme =  heatmap_colorscheme_load(cs[cs_enum]->data,cs[cs_enum]->size);

    heatmap_render_to(*hm,scheme,&image[0]);

    add_bg(image, w,h, bg_clr);
    save_png(output_file,image,w,h);
    free(image);
    heatmap_colorscheme_free(scheme);

    return 0;
}



