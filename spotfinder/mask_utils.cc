/**
 * @file mask_utils.cc
 * @brief Implementation of mask upload and resolution filtering utilities.
 */
#include "mask_utils.hpp"

void apply_resolution_filtering(PitchedMalloc<uint8_t> mask,
                                int width,
                                int height,
                                float wavelength,
                                detector_geometry detector,
                                float dmin,
                                float dmax,
                                cudaStream_t stream) {
    // Define the block size and grid size for the kernel
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Set the parameters for the resolution mask kernel
    ResolutionMaskParams params{.mask_pitch = mask.pitch,
                                .width = width,
                                .height = height,
                                .wavelength = wavelength,
                                .detector = detector,
                                .dmin = dmin,
                                .dmax = dmax};

    // Launch the kernel to apply resolution filtering
    call_apply_resolution_mask(
      numBlocks, threadsPerBlock, 0, stream, mask.get(), params);

    CUDA_CHECK(cudaStreamSynchronize(stream));
}
