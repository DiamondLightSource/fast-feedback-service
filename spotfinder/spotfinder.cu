/**
 * Basic Naive Kernel
 * 
 * Does spotfinding in-kernel, without in-depth performance tweaking.
 * 
 */

// #include <bitshuffle.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "kernels/erosion.cuh"
#include "kernels/thresholding.cuh"
#include "spotfinder.cuh"

namespace cg = cooperative_groups;

/**
 * @brief Device function for writing out debug information in PNG and TXT formats.
 *
 * This function writes the specified device data to both PNG and TXT files,
 * applying a pixel conversion function for PNG output and a condition function
 * for TXT output.
 *
 * @tparam PixelTransformFunc A callable object that takes a pixel value and returns a transformed value.
 * @tparam PixelConditionFunc A callable object that checks a condition on a pixel.
 * @param device_data Pointer to the device data.
 * @param pitch_bytes The pitch of the data in bytes.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param stream The CUDA stream.
 * @param filename The base filename for the output.
 * @param pixel_transform The function to transform pixel values for PNG output.
 * @param condition The function to check pixel conditions for TXT output.
 */
template <typename PixelTransformFunc, typename PixelConditionFunc>
void debug_writeout(uint8_t *device_data,
                    size_t pitch_bytes,
                    int width,
                    int height,
                    cudaStream_t stream,
                    const char *filename,
                    PixelTransformFunc pixel_transform,
                    PixelConditionFunc condition) {
    // Write to PNG using the pixel transformation function
    save_device_data_to_png(
      device_data, pitch_bytes, width, height, stream, filename, pixel_transform);

    // Write to TXT using the condition function
    save_device_data_to_txt(
      device_data, pitch_bytes, width, height, stream, filename, condition);
}

#pragma region Launch Wrappers
/**
 * @brief Wrapper function to call the dispersion-based spotfinding algorithm.
 * This function launches the `compute_dispersion_threshold_kernel` to perform
 * the spotfinding based on the basic dispersion threshold.
 *
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param image PitchedMalloc object for the image data.
 * @param mask PitchedMalloc object for the mask data indicating valid pixels.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param result_strong (Output) Device pointer for the strong pixel mask data to be written to.
 * @param min_count The minimum number of valid pixels required in the local neighborhood. Default is 3.
 * @param n_sig_b The background noise significance level. Default is 6.0.
 * @param n_sig_s The signal significance level. Default is 3.0.
 */
void call_do_spotfinding_dispersion(dim3 blocks,
                                    dim3 threads,
                                    size_t shared_memory,
                                    cudaStream_t stream,
                                    PitchedMalloc<pixel_t> &image,
                                    PitchedMalloc<uint8_t> &mask,
                                    int width,
                                    int height,
                                    pixel_t max_valid_pixel_value,
                                    PitchedMalloc<uint8_t> *result_strong,
                                    uint8_t min_count,
                                    float n_sig_b,
                                    float n_sig_s) {
    /// One-direction width of kernel. Total kernel span is (K * 2 + 1)
    constexpr uint8_t basic_kernel_radius = 3;

    // Launch the dispersion threshold kernel
    compute_threshold_kernel<<<blocks, threads, shared_memory, stream>>>(
      image.get(),            // Image data pointer
      mask.get(),             // Mask data pointer
      result_strong->get(),   // Output mask pointer
      image.pitch,            // Image pitch
      mask.pitch,             // Mask pitch
      result_strong->pitch,   // Output mask pitch
      width,                  // Image width
      height,                 // Image height
      max_valid_pixel_value,  // Maximum valid pixel value
      basic_kernel_radius,    // Kernel width
      basic_kernel_radius,    // Kernel height
      min_count,              // Minimum count
      n_sig_b,                // Background significance level
      n_sig_s                 // Signal significance level
    );

    cudaStreamSynchronize(
      stream);  // Synchronize the CUDA stream to ensure the kernel is complete
}

/**
 * @brief Wrapper function to call the extended dispersion-based spotfinding algorithm.
 * This function launches the `compute_final_threshold_kernel` for final thresholding
 * after applying the dispersion mask and the `compute_dispersion_threshold_kernel`
 * for initial thresholding.
 *
 * @param blocks The dimensions of the grid of blocks.
 * @param threads The dimensions of the grid of threads within each block.
 * @param shared_memory The size of shared memory required per block (in bytes).
 * @param stream The CUDA stream to execute the kernel.
 * @param image PitchedMalloc object for the image data.
 * @param mask PitchedMalloc object for the mask data indicating valid pixels.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param max_valid_pixel_value The maximum valid trusted pixel value.
 * @param result_strong (Output) Device pointer for the strong pixel mask data to be written to.
 * @param do_writeout Flag to indicate if the output should be written to file. Default is false.
 * @param min_count The minimum number of valid pixels required in the local neighborhood. Default is 3.
 * @param n_sig_b The background noise significance level. Default is 6.0.
 * @param n_sig_s The signal significance level. Default is 3.0.
 * @param threshold The global threshold for intensity values. Default is 10.0.
 */
void call_do_spotfinding_extended(dim3 blocks,
                                  dim3 threads,
                                  size_t shared_memory,
                                  cudaStream_t stream,
                                  PitchedMalloc<pixel_t> &image,
                                  PitchedMalloc<uint8_t> &mask,
                                  int width,
                                  int height,
                                  pixel_t max_valid_pixel_value,
                                  PitchedMalloc<uint8_t> *result_strong,
                                  bool do_writeout,
                                  uint8_t min_count,
                                  float n_sig_b,
                                  float n_sig_s,
                                  float threshold) {
    // Allocate intermediate buffer for the dispersion mask on the device
    PitchedMalloc<uint8_t> d_dispersion_mask(width, height);

    constexpr uint8_t first_pass_kernel_radius = 3;

    /*
     * First pass
     * Perform the initial dispersion thresholding only on the background
     * threshold. The surviving pixels are then used as a mask later to
     * exclude them from the background calculation in the second pass.
    */
    {
        // First pass 🔎 Perform the initial dispersion thresholding
        compute_dispersion_threshold_kernel<<<blocks, threads, shared_memory, stream>>>(
          image.get(),               // Image data pointer
          mask.get(),                // Mask data pointer
          d_dispersion_mask.get(),   // Output dispersion mask pointer
          image.pitch,               // Image pitch
          mask.pitch,                // Mask pitch
          d_dispersion_mask.pitch,   // Output dispersion mask pitch
          width,                     // Image width
          height,                    // Image height
          max_valid_pixel_value,     // Maximum valid pixel value
          first_pass_kernel_radius,  // Kernel radius
          first_pass_kernel_radius,  // Kernel radius
          min_count,                 // Minimum count
          n_sig_b,                   // Background significance level
          n_sig_s                    // Signal significance level
        );
        cudaStreamSynchronize(
          stream);  // Synchronize the CUDA stream to ensure the first pass is complete

        if (do_writeout) {
            printf("First pass complete\n");
            debug_writeout(
              d_dispersion_mask.get(),
              d_dispersion_mask.pitch_bytes(),
              width,
              height,
              stream,
              "first_pass_dispersion_result",
              [](uint8_t pixel) { return pixel == MASKED_PIXEL ? 0 : 255; },
              [](uint8_t pixel) { return pixel != 0; });
        }
    }

    /*
     * Erosion pass ✂
     * Erode the first pass results.
     * The surviving pixels are then used as a mask to exclude them
     * from the background calculation in the second pass.
    */
    PitchedMalloc<uint8_t> d_erosion_mask(width, height);

    {  // Scope for erosion pass launch parameters
        // dim3 threads_per_erosion_block(32, 32);
        // dim3 erosion_blocks(
        //   (width + threads_per_erosion_block.x - 1) / threads_per_erosion_block.x,
        //   (height + threads_per_erosion_block.y - 1) / threads_per_erosion_block.y);

        // Calculate the shared memory size for the erosion kernel
        // size_t erosion_shared_memory =
        //   (threads_per_erosion_block.x + 2 * first_pass_kernel_radius)
        //   * (threads_per_erosion_block.y + 2 * first_pass_kernel_radius)
        //   * sizeof(uint8_t);

        // Perform erosion
        erosion_kernel<<<blocks, threads, shared_memory, stream>>>(
          d_dispersion_mask.get(),
          d_erosion_mask.get(),
          mask.get(),
          d_dispersion_mask.pitch,
          d_erosion_mask.pitch,
          mask.pitch,
          width,
          height,
          first_pass_kernel_radius);
        cudaStreamSynchronize(
          stream);  // Synchronize the CUDA stream to ensure the erosion pass is complete

        if (do_writeout) {
            printf("Erosion pass complete\n");
            debug_writeout(
              d_erosion_mask.get(),
              d_erosion_mask.pitch_bytes(),
              width,
              height,
              stream,
              "eroded_dispersion_result",
              [](uint8_t pixel) { return pixel == MASKED_PIXEL ? 0 : 255; },
              [](uint8_t pixel) { return pixel == MASKED_PIXEL; });
        }
    }

    constexpr uint8_t second_pass_kernel_radius = 5;

    /*
     * Second pass 🎯
     * Perform the final thresholding using the dispersion mask.
    */
    {
        printf("Second pass\n");
        // Second pass: Perform the final thresholding using the dispersion mask
        compute_final_threshold_kernel<<<blocks, threads, shared_memory, stream>>>(
          image.get(),                // Image data pointer
          mask.get(),                 // Mask data pointer
          d_erosion_mask.get(),       // Dispersion mask pointer
          result_strong->get(),       // Output result mask pointer
          image.pitch,                // Image pitch
          mask.pitch,                 // Mask pitch
          d_erosion_mask.pitch,       // Dispersion mask pitch
          result_strong->pitch,       // Output result mask pitch
          width,                      // Image width
          height,                     // Image height
          max_valid_pixel_value,      // Maximum valid pixel value
          second_pass_kernel_radius,  // Kernel radius
          second_pass_kernel_radius,  // Kernel radius
          n_sig_s,                    // Signal significance level
          threshold                   // Global threshold
        );
        cudaStreamSynchronize(
          stream);  // Synchronize the CUDA stream to ensure the second pass is complete

        if (do_writeout) {
            printf("Second pass complete\n");
            debug_writeout(
              result_strong->get(),
              mask.pitch_bytes(),
              width,
              height,
              stream,
              "final_extended_threshold_result",
              [](uint8_t pixel) { return pixel == VALID_PIXEL ? 255 : 0; },
              [](uint8_t pixel) { return pixel != 0; });
        }
    }
}

#pragma endregion Launch Wrappers
