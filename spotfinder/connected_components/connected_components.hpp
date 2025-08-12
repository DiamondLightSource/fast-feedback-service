#ifndef CONNECTED_COMPONENTS_HPP
#define CONNECTED_COMPONENTS_HPP

#include <builtin_types.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <cstdint>
#include <vector>

#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"

struct Signal {
    int x, y;              // Coordinates of the pixel
    std::optional<int> z;  // Optional z-index used in 3DCC
    pixel_t intensity;     // Pixel intensity
    size_t linear_index;   // Linear index of the pixel
};

struct Reflection {
    int l, t, r, b;  // Bounding box: left, top, right, bottom
    int num_pixels = 0;
};

class Reflection3D {
  public:
    Reflection3D()
        : x_min_(std::numeric_limits<int>::max()),
          x_max_(std::numeric_limits<int>::min()),
          y_min_(std::numeric_limits<int>::max()),
          y_max_(std::numeric_limits<int>::min()),
          z_min_(std::numeric_limits<int>::max()),
          z_max_(std::numeric_limits<int>::min()),
          num_pixels_(0),
          com_cached_(false) {}

    void add_signal(const Signal &signal) {
        signals_.push_back(signal);

        // Invalidate cached center of mass
        com_cached_ = false;

        // Update bounding box
        if (signal.z.has_value()) {
            z_min_ = std::min(z_min_, signal.z.value());
            z_max_ = std::max(z_max_, signal.z.value());
        } else {
            std::string msg = "Signal missing z-index";
            logger.error(msg);
            throw std::runtime_error(msg);
        }
        x_min_ = std::min(x_min_, signal.x);
        x_max_ = std::max(x_max_, signal.x);
        y_min_ = std::min(y_min_, signal.y);
        y_max_ = std::max(y_max_, signal.y);

        ++num_pixels_;  // Increment the number of pixels in the reflection
    }

    /**
     * @brief Calculate or retrieve cached center of mass of the 3D reflection.
     * 
     * @return A tuple containing the x, y, and z coordinates of the center of mass.
     */
    std::tuple<float, float, float> center_of_mass() const {
        if (com_cached_) {
            return com_cache_;
        }

        if (signals_.empty()) {
            logger.error("No pixels in 3D reflection");
            throw std::runtime_error("No pixels in 3D reflection");
        }

        double weighted_sum_x = 0, weighted_sum_y = 0, weighted_sum_z = 0;
        double total_intensity = 0;

        for (const auto &signal : signals_) {
            weighted_sum_x += (signal.x + 0.5) * signal.intensity;
            weighted_sum_y += (signal.y + 0.5) * signal.intensity;
            weighted_sum_z += (signal.z.value() + 0.5) * signal.intensity;
            total_intensity += signal.intensity;
        }

        if (total_intensity == 0) {
            logger.error("Total intensity is zero");
            throw std::runtime_error("Total intensity is zero");
        }

        // Compute and cache the result
        com_cache_ = {weighted_sum_x / total_intensity,
                      weighted_sum_y / total_intensity,
                      weighted_sum_z / total_intensity};

        com_cached_ = true;  // Mark cache as valid
        return com_cache_;
    }

    /**
     * @brief Calculate the distance between the pixel with the highest intensity
     *        and the center of mass.
     * 
     * @return The Euclidean distance between the peak pixel and the center of mass.
     */
    float peak_centroid_distance() const {
        if (signals_.empty()) {
            logger.error("No pixels in 3D reflection");
            throw std::runtime_error("No pixels in 3D reflection");
        }

        // Find the signal with the highest intensity
        const Signal *peak_signal = nullptr;
        double max_intensity = std::numeric_limits<double>::min();

        for (const auto &signal : signals_) {
            if (signal.intensity > max_intensity) {
                max_intensity = signal.intensity;
                peak_signal = &signal;
            }
        }

        if (!peak_signal) {
            logger.error("Failed to find peak intensity signal");
            throw std::runtime_error("Failed to find peak intensity signal");
        }

        // Get the cached or computed center of mass
        auto [com_x, com_y, com_z] = center_of_mass();

        // Calculate the Euclidean distance
        float dx = (peak_signal->x + 0.5f) - com_x;
        float dy = (peak_signal->y + 0.5f) - com_y;
        float dz = (peak_signal->z.value() + 0.5f) - com_z;

        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    // Getters for bounding box
    inline int get_x_min() const {
        return x_min_;
    }
    inline int get_x_max() const {
        return x_max_;
    }
    inline int get_y_min() const {
        return y_min_;
    }
    inline int get_y_max() const {
        return y_max_;
    }
    inline int get_z_min() const {
        return z_min_;
    }
    inline int get_z_max() const {
        return z_max_;
    }
    inline int get_num_pixels() const {
        return num_pixels_;
    }

  private:
    std::vector<Signal> signals_;
    int x_min_, x_max_;
    int y_min_, y_max_;
    int z_min_, z_max_;
    int num_pixels_;

    // com cache for lazy evaluation
    mutable bool com_cached_;
    mutable std::tuple<float, float, float> com_cache_;
};

/**
 * @brief Class to find connected components in a 2D image
 * 
 * The `ConnectedComponents` class identifies connected components (clusters) in a
 * binary 2D image. It stores signal pixels and their corresponding graph structure
 * to determine connected regions.
 * 
 * @param result_image the result image
 * @param original_image the original image
 * @param width the width of the image
 * @param height the height of the image
 * @param z_index the z-index of the image
 * @param min_spot_size the minimum spot size
 * 
 * @note `result_image` and `original_image` are only used to construct the
 * signals and are not stored in the object, nor are they used again once
 * initialization is complete.
 */
class ConnectedComponents {
  public:
    ConnectedComponents(const uint8_t *result_image,
                        const pixel_t *original_image,
                        const ushort width,
                        const ushort height,
                        const uint min_spot_size);

    uint get_num_strong_pixels() const {
        return num_strong_pixels;
    }

    uint get_num_strong_pixels_filtered() const {
        return num_strong_pixels_filtered;
    }

    /*
     * By returning const references, we prevent the caller from 
     * modifying the internal maps, but allow them to read the data
     * without a costly copy.
    */

    const std::vector<Reflection> &get_boxes() const {
        return boxes;
    }

    std::unordered_map<size_t, Signal> &get_signals() {
        return signals;
    }

    const auto &get_graph() const {
        return graph;
    }

    const auto &get_index_to_vertex() const {
        return index_to_vertex;
    }

    const auto &get_vertex_to_index() const {
        return vertex_to_index;
    }

    /**
   * @brief Finds 3D connected components using a list of ConnectedComponents objects.
   *
   * This function combines the precomputed 2D graphs and signals stored in
   * ConnectedComponents objects into a single 3D graph. It adds inter-slice
   * (z-axis) connectivity and computes connected components across slices.
   *
   * @param slices A list of ConnectedComponents objects, each corresponding to a slice.
   * @param width The width of the image.
   * @param height The height of the image.
   * @param min_spot_size The minimum number of pixels in a connected component.
   * @param max_peak_centroid_separation The maximum distance between peak and centroid.
   * @return A list of 3D reflections.
   */
    static std::vector<Reflection3D> find_3d_components(
      const std::vector<std::unique_ptr<ConnectedComponents>> &slices,
      const ushort width,
      const ushort height,
      const uint min_spot_size,
      const float max_peak_centroid_separation);

    std::vector<Reflection3D> find_2d_components(
      const uint min_spot_size,
      const float max_peak_centroid_separation);

  private:
    uint num_strong_pixels;           // Number of strong pixels
    uint num_strong_pixels_filtered;  // Number of strong pixels after filtering
    std::vector<Reflection> boxes;    // Bounding boxes
    // Maps pixel linear index -> Signal (used to store signal pixels)
    std::unordered_map<size_t, Signal> signals;
    // Maps graph linear index -> vertex ID
    std::unordered_map<size_t, size_t> index_to_vertex;
    // Maps graph vertex ID -> linear index
    std::unordered_map<size_t, size_t> vertex_to_index;
    // 2D graph representing the connected components
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph;

    /**
     * @brief Builds a graph where vertices represent signal pixels and edges represent 
     * connectivity between neighboring pixels.
     * 
     * This function uses the `signals` map to find neighboring pixels efficiently.
     * The `index_to_vertex` is used to map linear indices (from `signals`) to graph vertex IDs.
     */
    void build_graph(const ushort width, const ushort height);

    /**
     * @brief Generates bounding boxes for connected components using the graph labels.
     * 
     * The `labels` vector maps each graph vertex to its connected component ID.
     * The `index_to_vertex` map is used to map linear indices to graph vertex IDs.
     */
    void generate_boxes(const ushort width,
                        const ushort height,
                        const uint32_t min_spot_size);
};

#endif  // CONNECTED_COMPONENTS_HPP