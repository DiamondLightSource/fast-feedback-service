#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "../spotfinder.cuh"
#include "thresholding.cuh"

namespace cg = cooperative_groups;

#pragma region Device Functions
/*
 * TODO: Break out redundant code from the thresholding kernels into device functions.
*/
#pragma endregion

#pragma region Thresholding kernels
/**
 * @brief Kernel for computing the basic threshold based on variance and mean.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param result_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param threshold The global threshold for intensity.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 */
__global__ void compute_threshold_kernel(pixel_t *image,
                                         uint8_t *mask,
                                         uint8_t *result_mask,
                                         size_t image_pitch,
                                         size_t mask_pitch,
                                         size_t result_pitch,
                                         int width,
                                         int height,
                                         pixel_t max_valid_pixel_value,
                                         int kernel_width,
                                         int kernel_height,
                                         int min_count,
                                         float n_sig_b,
                                         float n_sig_s) {
    // Move pointers to the correct slice
    image = image + (image_pitch * height * blockIdx.z);
    result_mask = result_mask + (mask_pitch * height * blockIdx.z);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (x >= width || y >= height) return;  // Out of bounds guard

    pixel_t this_pixel = image[y * image_pitch + x];

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid =
      mask[y * mask_pitch + x] != 0 && this_pixel <= max_valid_pixel_value;

    // Initialize variables for computing the local sum and count
    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

    int row_start = max(0, y - kernel_height);
    int row_end = min(y + kernel_height + 1, height);

    for (int row = row_start; row < row_end; ++row) {
        int row_offset = image_pitch * row;
        int mask_offset = mask_pitch * row;

        int col_start = max(0, x - kernel_width);
        int col_end = min(x + kernel_width + 1, width);

        for (int col = col_start; col < col_end; ++col) {
            pixel_t pixel = image[row_offset + col];
            uint8_t mask_pixel = mask[mask_offset + col];
            bool include_pixel = mask_pixel != 0;  // If the pixel is valid
            if (include_pixel) {
                sum += pixel;
                sumsq += pixel * pixel;
                n += 1;
            }
        }
    }

    if (px_is_valid && n > 1) {
        // Compute local mean and variance
        float sum_f = static_cast<float>(sum);
        float sumsq_f = static_cast<float>(sumsq);

        float mean = sum_f / n;
        float variance = (n * sumsq_f - (sum_f * sum_f)) / (n * (n - 1));
        float dispersion = variance / mean;

        // Compute the background threshold and signal threshold
        float background_threshold = 1 + n_sig_b * sqrt(2.0f / (n - 1));
        bool not_background = dispersion > background_threshold;
        float signal_threshold = mean + n_sig_s * sqrt(mean);

        // Check if the pixel is a strong pixel
        bool is_signal = this_pixel > signal_threshold;

        result_mask[x + result_pitch * y] = not_background && is_signal;
    } else {
        result_mask[x + result_pitch * y] = 0;
    }
}

/**
 * @brief Kernel for computing the dispersion threshold.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param result_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param kernel_width The radius of the kernel in the x-direction.
 * @param kernel_height The radius of the kernel in the y-direction.
 * @param min_count Minimum number of valid pixels required to be considered a valid spot.
 * @param n_sig_b Background noise significance level.
 * @param n_sig_s Signal significance level.
 */
__global__ void compute_dispersion_threshold_kernel(pixel_t *image,
                                                    uint8_t *mask,
                                                    uint8_t *result_mask,
                                                    size_t image_pitch,
                                                    size_t mask_pitch,
                                                    size_t result_pitch,
                                                    int width,
                                                    int height,
                                                    pixel_t max_valid_pixel_value,
                                                    int kernel_width,
                                                    int kernel_height,
                                                    int min_count,
                                                    float n_sig_b,
                                                    float n_sig_s) {
    // Move pointers to the correct slice
    image = image + (image_pitch * height * blockIdx.z);
    result_mask = result_mask + (mask_pitch * height * blockIdx.z);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (x >= width || y >= height) return;  // Out of bounds guard

    // if it is the first pixel in the image print the kernel width and height
    if (x == 0 && y == 0) {
        printf("Kernel width: %d, Kernel height: %d\n", kernel_width, kernel_height);
        printf("Block Idx.z: %d\n", blockIdx.z);
    }

    pixel_t this_pixel = image[y * image_pitch + x];

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid =
      mask[y * mask_pitch + x] != 0 && this_pixel <= max_valid_pixel_value;

    // Initialize variables for computing the local sum and count
    uint sum = 0;
    size_t sumsq = 0;
    uint8_t n = 0;

    int row_start = max(0, y - kernel_height);
    int row_end = min(y + kernel_height + 1, height);

    for (int row = row_start; row < row_end; ++row) {
        int row_offset = image_pitch * row;
        int mask_offset = mask_pitch * row;

        int col_start = max(0, x - kernel_width);
        int col_end = min(x + kernel_width + 1, width);

        for (int col = col_start; col < col_end; ++col) {
            pixel_t pixel = image[row_offset + col];
            uint8_t mask_pixel = mask[mask_offset + col];
            bool include_pixel = mask_pixel != 0;  // If the pixel is valid
            if (include_pixel) {
                sum += pixel;
                sumsq += pixel * pixel;
                n += 1;
            }
        }
    }

    // Calculate the thresholding
    if (px_is_valid && n > 1) {
        // Compute local mean and variance
        float sum_f = static_cast<float>(sum);
        float sumsq_f = static_cast<float>(sumsq);

        float mean = sum_f / n;
        float variance = (n * sumsq_f - (sum_f * sum_f)) / (n * (n - 1));
        float dispersion = variance / mean;

        // Compute the background threshold
        float background_threshold = 1 + n_sig_b * sqrt(2.0f / (n - 1));
        bool not_background = dispersion > background_threshold;

        result_mask[x + result_pitch * y] = not_background;
    } else {
        result_mask[x + result_pitch * y] = 0;
    }
}

/**
 * @brief Kernel for computing the final threshold after dispersion mask.
 * @param image Pointer to the input image data.
 * @param mask Pointer to the mask data indicating valid pixels.
 * @param dispersion_mask Pointer to the dispersion mask used for extended algorithm.
 * @param result_mask Pointer to the output mask data where results will be stored.
 * @param image_pitch The pitch of the image data.
 * @param mask_pitch The pitch of the mask data.
 * @param dispersion_mask_pitch The pitch of the dispersion mask data.
 * @param result_mask_pitch The pitch of the result mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param n_sig_s Signal significance level.
 * @param threshold Global threshold for the intensity.
 */
__global__ void compute_final_threshold_kernel(pixel_t *image,
                                               uint8_t *mask,
                                               uint8_t *dispersion_mask,
                                               uint8_t *result_mask,
                                               size_t image_pitch,
                                               size_t mask_pitch,
                                               size_t dispersion_mask_pitch,
                                               size_t result_mask_pitch,
                                               int width,
                                               int height,
                                               pixel_t max_valid_pixel_value,
                                               int kernel_width,
                                               int kernel_height,
                                               float n_sig_s,
                                               float threshold) {
    // Move pointers to the correct slice
    image = image + (image_pitch * height * blockIdx.z);
    result_mask = result_mask + (mask_pitch * height * blockIdx.z);

    // Calculate the pixel coordinates
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x + block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y + block.thread_index().y;

    if (x >= width || y >= height) return;  // Out of bounds guard

    pixel_t this_pixel = image[y * image_pitch + x];

    // Check if the pixel is masked and below the maximum valid pixel value
    bool px_is_valid =
      mask[y * mask_pitch + x] != 0 && this_pixel <= max_valid_pixel_value;

    // Initialize variables for computing the local sum and count
    uint sum = 0;
    uint8_t n = 0;

    int row_start = max(0, y - kernel_height);
    int row_end = min(y + kernel_height + 1, height);

    for (int row = row_start; row < row_end; ++row) {
        int row_offset = image_pitch * row;
        int mask_offset = mask_pitch * row;

        int col_start = max(0, x - kernel_width);
        int col_end = min(x + kernel_width + 1, width);

        for (int col = col_start; col < col_end; ++col) {
            pixel_t pixel = image[row_offset + col];
            uint8_t mask_pixel = mask[mask_offset + col];
            uint8_t disp_mask_pixel =
              dispersion_mask[row * dispersion_mask_pitch + col];
            bool include_pixel =
              mask_pixel != 0 && disp_mask_pixel;  // If the pixel is valid
            if (include_pixel) {
                sum += pixel;
                n += 1;
            }
        }
    }

    // Calculate the thresholding
    if (px_is_valid && n > 1) {
        float sum_f = static_cast<float>(sum);

        bool disp_mask = !dispersion_mask[y * dispersion_mask_pitch + x];
        bool global_mask = image[y * image_pitch + x] > threshold;
        float mean = (n > 1 ? sum_f / n : 0);  // If n is less than 1, set mean to 0
        bool local_mask = image[y * image_pitch + x] >= (mean + n_sig_s * sqrtf(mean));

        result_mask[y * result_mask_pitch + x] = disp_mask && global_mask && local_mask;
    } else {
        result_mask[y * result_mask_pitch + x] = 0;
    }
}
#pragma endregion