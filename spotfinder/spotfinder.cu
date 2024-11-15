/**
 * Basic Naive Kernel
 * 
 * Does spotfinding in-kernel, without in-depth performance tweaking.
 * 
 */

// #include <bitshuffle.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "device_common.cuh"
#include "kernels/erosion.cuh"
#include "kernels/thresholding.cuh"
#include "spotfinder.cuh"

namespace cg = cooperative_groups;

__constant__ KernelConstants kernel_constants;

/**
 * @brief Set the kernel constants in the device constant memory.
 * 
 * @warning This function is asynchronous and does not guarantee that
 * the kernel constants are set before the next kernel call. It is the
 * responsibility of the caller to ensure that the kernel constants are
 * set and synchronized before the next kernel call.
 */
void set_kernel_constants(cudaStream_t stream,
                          size_t image_pitch,
                          size_t mask_pitch,
                          size_t result_pitch,
                          ushort width,
                          ushort height,
                          float max_valid_pixel_value,
                          uint8_t min_count,
                          float n_sig_b,
                          float n_sig_s) {
    KernelConstants host_constants{
      image_pitch,
      mask_pitch,
      result_pitch,
      width,
      height,
      max_valid_pixel_value,
      min_count,
      n_sig_b,
      n_sig_s,
    };

    CUDA_CHECK(cudaMemcpyToSymbolAsync(kernel_constants,
                                       &host_constants,
                                       sizeof(KernelConstants),
                                       0,
                                       cudaMemcpyHostToDevice,
                                       stream));
}

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
 * This function launches the `dispersion` kernel to perform
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
    // Set the kernel constants in the device constant memory
    set_kernel_constants(stream,
                         image.pitch,
                         mask.pitch,
                         result_strong->pitch,
                         width,
                         height,
                         max_valid_pixel_value,
                         min_count,
                         n_sig_b,
                         n_sig_s);

    // Synchronize the CUDA stream to ensure the kernel constants are set
    cudaStreamSynchronize(stream);

    // Launch the dispersion threshold kernel
    dispersion<<<blocks, threads, shared_memory, stream>>>(
      image.get(),          // Image data pointer
      mask.get(),           // Mask data pointer
      result_strong->get()  // Output mask pointer
    );

    cudaStreamSynchronize(
      stream);  // Synchronize the CUDA stream to ensure the kernel is complete
}

/**
 * @brief Wrapper function to call the extended dispersion-based spotfinding algorithm.
 * This function launches the `dispersion_extended_second_pass` for final thresholding
 * after applying the dispersion mask and the `dispersion_extended_first_pass`
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
 * @param threshold The global threshold for intensity values. Default is 0.
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
                                  float n_sig_s) {
    // Set the kernel constants in the device constant memory
    set_kernel_constants(stream,
                         image.pitch,
                         mask.pitch,
                         result_strong->pitch,
                         width,
                         height,
                         max_valid_pixel_value,
                         min_count,
                         n_sig_b,
                         n_sig_s);

    // Allocate intermediate buffer for the dispersion mask on the device
    PitchedMalloc<uint8_t> d_dispersion_mask(width, height);
    PitchedMalloc<uint8_t> d_erosion_mask(width, height);

    // Synchronize the CUDA stream to ensure the kernel constants are set
    cudaStreamSynchronize(stream);

    /*
     * First pass 🔎
     * Perform the initial dispersion thresholding only on the background
     * threshold. The surviving pixels are then used as a mask later to
     * exclude them from the background calculation in the second pass.
    */
    dispersion_extended_first_pass<<<blocks, threads, shared_memory, stream>>>(
      image.get(),             // Image data pointer
      mask.get(),              // Mask data pointer
      d_dispersion_mask.get()  // Output dispersion mask pointer
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

    /*
     * Erosion pass ✂
     * Erode the first pass results.
     * The surviving pixels are then used as a mask to exclude them
     * from the background calculation in the second pass.
    */

    // Perform erosion
    erosion<<<blocks, threads, shared_memory, stream>>>(d_dispersion_mask.get(),
                                                        d_erosion_mask.get(),
                                                        d_dispersion_mask.pitch,
                                                        d_erosion_mask.pitch,
                                                        KERNEL_RADIUS);
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

    /*
     * Second pass 🎯
     * Perform the final thresholding using the dispersion mask.
    */
    dispersion_extended_second_pass<<<blocks, threads, shared_memory, stream>>>(
      image.get(),           // Image data pointer
      mask.get(),            // Mask data pointer
      d_erosion_mask.get(),  // Dispersion mask pointer
      result_strong->get(),  // Output result mask pointer
      d_erosion_mask.pitch   // Dispersion mask pitch
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

#pragma endregion Launch Wrappers
