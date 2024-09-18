#ifndef SPOTFINDER_H
#define SPOTFINDER_H

#include <builtin_types.h>

#include <nlohmann/json.hpp>

#include "common.hpp"
#include "cuda_common.hpp"
#include "h5read.h"

#define VALID_PIXEL 1
#define MASKED_PIXEL 0

using pixel_t = H5Read::image_type;

/**
 * @brief Struct to store the geometry of the detector.
 * @param pixel_size_x The pixel size of the detector in the x-direction in mm.
 * @param pixel_size_y The pixel size of the detector in the y-direction in mm.
 * @param beam_center_x The x-coordinate of the beam center in the image.
 * @param beam_center_y The y-coordinate of the beam center in the image.
 * @param distanc The distance from the sample to the detector in mm.
*/
struct detector_geometry {
    float pixel_size_x;
    float pixel_size_y;
    float beam_center_x;
    float beam_center_y;
    float distance;

    /**
     * @brief Default constructor for detector_geometry.
     * Initializes the members with zeroed values.
     */
    detector_geometry()
        : pixel_size_x(0.0f),
          pixel_size_y(0.0f),
          beam_center_x(0.0f),
          beam_center_y(0.0f),
          distance(0.0f) {}

    /**
     * @brief Constructor to initialize the detector geometry from a JSON object.
     * @param geometry_data A JSON object containing the detector geometry data.
     * The JSON object must have the following keys:
     * - pixel_size_x: The pixel size of the detector in the x-direction in mm
     * - pixel_size_y: The pixel size of the detector in the y-direction in mm
     * - beam_center_x: The x-coordinate of the pixel beam center in the image
     * - beam_center_y: The y-coordinate of the pixel beam center in the image
     * - distance: The distance from the sample to the detector in mm
    */
    detector_geometry(nlohmann::json geometry_data) {
        std::vector<std::string> required_keys = {
          "pixel_size_x", "pixel_size_y", "beam_center_x", "beam_center_y", "distance"};

        for (const auto &key : required_keys) {
            if (geometry_data.find(key) == geometry_data.end()) {
                throw std::invalid_argument("Key " + key
                                            + " is missing from the input JSON");
            }
        }

        pixel_size_x = geometry_data["pixel_size_x"];
        pixel_size_y = geometry_data["pixel_size_y"];
        beam_center_x = geometry_data["beam_center_x"];
        beam_center_y = geometry_data["beam_center_y"];
        distance = geometry_data["distance"];
    }
};

/**
 * @brief Struct to store parameters for calculating the resolution filtered mask
 * @param mask_pitch The pitch (width in bytes) of the mask data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param wavelength The wavelength of the X-ray beam in Ångströms.
 * @param detector The geometry of the detector.
 * @param dmin The minimum resolution (d-spacing) threshold.
 * @param dmax The maximum resolution (d-spacing) threshold.
*/
struct ResolutionMaskParams {
    size_t mask_pitch;
    int width;
    int height;
    float wavelength;
    detector_geometry detector;
    float dmin;
    float dmax;
};

void call_apply_resolution_mask(dim3 blocks,
                                dim3 threads,
                                size_t shared_memory,
                                cudaStream_t stream,
                                uint8_t *mask,
                                ResolutionMaskParams params);

void call_do_spotfinding_dispersion(dim3 blocks,
                                    dim3 threads,
                                    size_t shared_memory,
                                    cudaStream_t stream,
                                    PitchedMalloc<pixel_t> &image,
                                    PitchedMalloc<uint8_t> &mask,
                                    int width,
                                    int height,
                                    pixel_t max_valid_pixel_value,
                                    uint8_t *result_strong);

void call_do_spotfinding_extended(dim3 blocks,
                                  dim3 threads,
                                  size_t shared_memory,
                                  cudaStream_t stream,
                                  PitchedMalloc<pixel_t> &image,
                                  PitchedMalloc<uint8_t> &mask,
                                  int width,
                                  int height,
                                  pixel_t max_valid_pixel_value,
                                  uint8_t *result_strong,
                                  bool do_writeout = false);

#endif