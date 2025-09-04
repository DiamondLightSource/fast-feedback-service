#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/os.h>
#include <lodepng.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "common.hpp"

// #if __has_include(<hdf5.h>)
// #define HAS_HDF5
// namespace _hdf5 {
// #include <hdf5.h>
// }
// #endif

class cuda_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

inline auto cuda_error_string(cudaError_t err) {
    const char *err_name = cudaGetErrorName(err);
    const char *err_str = cudaGetErrorString(err);
    return fmt::format("{}: {}", std::string{err_name}, std::string{err_str});
}
inline auto _cuda_check_error(cudaError_t err, const char *file, int line_num) {
    if (err != cudaSuccess) {
        throw cuda_error(
          fmt::format("{}:{}: {}", file, line_num, cuda_error_string(err)));
    }
}

#define CUDA_CHECK(x) _cuda_check_error((x), __FILE__, __LINE__)

/// Raise an exception IF CUDA is in an error state, with the name and description
inline auto cuda_throw_error() -> void {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw cuda_error(cuda_error_string(err));
    }
}

template <typename T>
auto make_cuda_malloc(size_t num_items = 1) {
    using Tb = typename std::remove_extent<T>::type;
    Tb *obj = nullptr;
    auto err = cudaMalloc(&obj, sizeof(Tb) * num_items);
    if (err != cudaSuccess || obj == nullptr) {
        throw cuda_error(
          fmt::format("Error in make_cuda_malloc: {}", cuda_error_string(err)));
    }
    auto deleter = [](Tb *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

template <typename T>
auto make_cuda_managed_malloc(size_t num_items) {
    using Tb = typename std::remove_extent<T>::type;
    Tb *obj = nullptr;
    auto err = cudaMallocManaged(&obj, sizeof(Tb) * num_items);
    if (err != cudaSuccess || obj == nullptr) {
        throw cuda_error(
          fmt::format("Error in make_cuda_managed_malloc: {}", cuda_error_string(err)));
    }

    auto deleter = [](Tb *ptr) { cudaFree(ptr); };
    return std::unique_ptr<T, decltype(deleter)>{obj, deleter};
}

/// Allocate memory using cudaMallocHost
template <typename T>
auto make_cuda_pinned_malloc(size_t num_items = 1) {
    using Tb = typename std::remove_extent<T>::type;
    Tb *obj = nullptr;
    auto err = cudaMallocHost(&obj, sizeof(Tb) * num_items);
    if (err != cudaSuccess || obj == nullptr) {
        throw cuda_error(
          fmt::format("Error in make_cuda_pinned_malloc: {}", cuda_error_string(err)));
    }
    auto deleter = [](Tb *ptr) { cudaFreeHost(ptr); };
    return std::shared_ptr<T[]>{obj, deleter};
}

template <typename T>
auto make_cuda_pitched_malloc(size_t width, size_t height) {
    static_assert(!std::is_unbounded_array_v<T>,
                  "T automatically returns unbounded array pointer");
    size_t pitch = 0;
    T *obj = nullptr;
    auto err = cudaMallocPitch(&obj, &pitch, width * sizeof(T), height);
    if (err != cudaSuccess || obj == nullptr) {
        throw cuda_error(
          fmt::format("Error in make_cuda_pitched_malloc: {}", cuda_error_string(err)));
    }

    auto deleter = [](T *ptr) { cudaFree(ptr); };

    return std::make_pair(std::shared_ptr<T[]>(obj, deleter), pitch / sizeof(T));
}

/**
 * @brief Function to allocate a pitched memory buffer on the GPU.
 * @param data The pointer to the allocated memory.
 * @param width The width of the buffer.
 * @param height The height of the buffer.
 * @param pitch The pitch of the buffer.
 */
template <typename T>
struct PitchedMalloc {
  public:
    using value_type = T;
    PitchedMalloc(std::shared_ptr<T[]> data, size_t width, size_t height, size_t pitch)
        : _data(data), width(width), height(height), pitch(pitch) {}

    PitchedMalloc(size_t width, size_t height) : width(width), height(height) {
        auto [alloc, alloc_pitch] = make_cuda_pitched_malloc<T>(width, height);
        _data = alloc;
        pitch = alloc_pitch;
    }

    auto get() {
        return _data.get();
    }
    auto size_bytes() -> size_t const {
        return pitch * height * sizeof(T);
    }
    auto pitch_bytes() -> size_t const {
        return pitch * sizeof(T);
    }

    std::shared_ptr<T[]> _data;
    size_t width;
    size_t height;
    size_t pitch;
};

class CudaStream {
    cudaStream_t _stream;

  public:
    CudaStream() {
        cudaStreamCreate(&_stream);
    }
    ~CudaStream() {
        cudaStreamDestroy(_stream);
    }
    operator cudaStream_t() const {
        return _stream;
    }
};

class CudaEvent {
    cudaEvent_t event;

  public:
    CudaEvent() {
        if (cudaEventCreate(&event) != cudaSuccess) {
            cuda_throw_error();
        }
    }
    CudaEvent(cudaEvent_t event) : event(event) {}

    ~CudaEvent() {
        cudaEventDestroy(event);
    }
    void record(cudaStream_t stream = 0) {
        if (cudaEventRecord(event, stream) != cudaSuccess) {
            cuda_throw_error();
        }
    }
    /// Elapsed Event time, in milliseconds
    float elapsed_time(CudaEvent &since) {
        float elapsed_time = 0.0f;
        if (cudaEventElapsedTime(&elapsed_time, since.event, event) != cudaSuccess) {
            cuda_throw_error();
        }
        return elapsed_time;
    }
    void synchronize() {
        if (cudaEventSynchronize(event) != cudaSuccess) {
            cuda_throw_error();
        }
    }
};

/**
 * @brief Save a 2D device array to a PNG file.
 *
 * @tparam PixelType The data type of the pixels (e.g., uint8_t).
 * @tparam TransformFunc A callable object or lambda that performs the pixel transformation.
 * @param device_ptr Pointer to the device array.
 * @param device_pitch The pitch (width in bytes) of the device array.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @param stream The CUDA stream to use for asynchronous copy and synchronization.
 * @param output_filename The name of the output PNG file.
 * @param transform_func The pixel transformation function (e.g., to invert pixel values).
 */
template <typename PixelType, typename TransformFunc>
void save_device_data_to_png(PixelType *device_ptr,
                             size_t device_pitch,
                             int width,
                             int height,
                             cudaStream_t stream,
                             const std::string &output_filename,
                             TransformFunc transform_func) {
    // Allocate host vector to hold the copied data
    std::vector<PixelType> host_data(width * height);

    // Copy data from device to host asynchronously
    cudaMemcpy2DAsync(host_data.data(),
                      width * sizeof(PixelType),  // Host pitch (bytes)
                      device_ptr,
                      device_pitch,               // Device pitch (bytes)
                      width * sizeof(PixelType),  // Width (bytes)
                      height,
                      cudaMemcpyDeviceToHost,
                      stream);

    // Synchronize the stream to ensure the copy is complete
    cudaStreamSynchronize(stream);

    // Apply the transformation function to each pixel
    for (auto &pixel : host_data) {
        pixel = transform_func(pixel);
    }

    // Encode and save the image as a PNG file
    lodepng::encode(fmt::format("{}.png", output_filename),
                    reinterpret_cast<const unsigned char *>(host_data.data()),
                    width,
                    height,
                    LCT_GREY);
}

/**
 * @brief Save the coordinates of pixels that satisfy a condition to a text file.
 *
 * @tparam PixelType The data type of the pixels (e.g., uint8_t).
 * @tparam ConditionFunc A callable object or lambda that determines which pixels should be logged.
 * @param device_ptr Pointer to the device array.
 * @param device_pitch The pitch (width in bytes) of the device array.
 * @param width The width of the image in pixels.
 * @param height The height of the image in pixels.
 * @param stream The CUDA stream to use for asynchronous copy and synchronization.
 * @param output_filename The name of the output text file.
 * @param condition_func The condition function that returns true for pixels to be logged.
 */
template <typename PixelType, typename ConditionFunc>
void save_device_data_to_txt(PixelType *device_ptr,
                             size_t device_pitch,
                             int width,
                             int height,
                             cudaStream_t stream,
                             const std::string &output_filename,
                             ConditionFunc condition_func) {
    // Allocate host vector to hold the copied data
    std::vector<PixelType> host_data(width * height);

    // Copy data from device to host asynchronously
    cudaMemcpy2DAsync(host_data.data(),
                      width * sizeof(PixelType),  // Host pitch (bytes)
                      device_ptr,
                      device_pitch,               // Device pitch (bytes)
                      width * sizeof(PixelType),  // Width (bytes)
                      height,
                      cudaMemcpyDeviceToHost,
                      stream);

    // Synchronize the stream to ensure the copy is complete
    cudaStreamSynchronize(stream);

    // Open an output file for the coordinates
    auto out = fmt::output_file(fmt::format("{}.txt", output_filename));

    // Write the coordinates of the pixels that satisfy the condition
    for (int y = 0, k = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++k) {
            if (condition_func(host_data[k])) {
                out.print("{}, {}, {}\n", x, y, host_data[k]);
            }
        }
    }
}

#endif