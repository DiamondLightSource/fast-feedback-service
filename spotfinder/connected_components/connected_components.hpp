#ifndef CONNECTED_COMPONENTS_HPP
#define CONNECTED_COMPONENTS_HPP

#include <builtin_types.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>
#include <dx2/detector.hpp>
#include <dx2/scan.hpp>

#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"

struct Signal {
    uint32_t x, y;         // Coordinates of the pixel
    std::optional<int> z;  // Optional z-index used in 3DCC
    pixel_t intensity;     // Pixel intensity
    size_t linear_index;   // Linear index of the pixel
};

struct Reflection {
    uint32_t l, t, r, b;  // Bounding box: left, top, right, bottom
    int num_pixels = 0;
};

class Reflection3D {
  public:
    Reflection3D()
        : x_min_(std::numeric_limits<uint32_t>::max()),
          x_max_(std::numeric_limits<uint32_t>::min()),
          y_min_(std::numeric_limits<uint32_t>::max()),
          y_max_(std::numeric_limits<uint32_t>::min()),
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

    std::tuple<double, double, int> kabsch_covariance(Vector3d& s1, Panel& panel, Vector3d& s0,
    Vector3d& m2, Scan& scan, double phi) const {
        Vector3d e1 = s1.cross(s0);
        e1.normalize();
        Vector3d e2 = s1.cross(e1);
        e2.normalize();
        double mags1 = std::sqrt(s1.dot(s1));
        double varx = 0;
        double vary = 0;
        double varz = 0;
        double total_intensity = 0;
        double zeta = m2.dot(e1);
        int image_range_0 = scan.get_image_range()[0];
        double oscillation_width = scan.get_oscillation()[1];
        double oscillation_start = scan.get_oscillation()[0];
        for (const auto &signal : signals_) {
            double x = static_cast<double>(signal.x) + 0.5;
            double y = static_cast<double>(signal.y) + 0.5;
            double z = static_cast<double>(signal.z.value()) + 0.5;
            auto [xmm, ymm] = panel.px_to_mm(x,y);
            Vector3d s1p = panel.get_lab_coord(xmm, ymm);
            Vector3d delta_s1 = s1p - s1;
            double eps1 = e1.dot(delta_s1) / mags1;
            double eps2 = e2.dot(delta_s1) / mags1;
            double phi_dash = (oscillation_start + (z - image_range_0) * oscillation_width)  * M_PI / 180.0;
            double eps3 = (phi_dash - phi) * zeta;
            varx += signal.intensity * eps1 * eps1;
            vary += signal.intensity * eps2 * eps2;
            varz += signal.intensity * eps3 * eps3;
            total_intensity += signal.intensity;
        }
        varx = varx / total_intensity;
        vary = vary / total_intensity;
        varz = varz / total_intensity;
        // Reason for dividing by two below, see https://github.com/dials/dials/issues/2851#issuecomment-2657018707
        return std::make_tuple((varx + vary) / 2.0, varz, z_max_ - z_min_);
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
            weighted_sum_x += (static_cast<double>(signal.x) + 0.5) * signal.intensity;
            weighted_sum_y += (static_cast<double>(signal.y) + 0.5) * signal.intensity;
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

        // logger.debug("Finding peak signal for reflection with {} pixels",
        //              signals_.size());

        // Find the signal with the highest intensity
        const Signal *peak_signal = nullptr;
        double max_intensity = std::numeric_limits<double>::min();
        int candidates_with_max_intensity = 0;

        for (const auto &signal : signals_) {
            // Guard: Skip if intensity is lower than current max
            if (signal.intensity < max_intensity) {
                continue;
            }

            // Handle new maximum intensity
            if (signal.intensity > max_intensity) {
                max_intensity = signal.intensity;
                peak_signal = &signal;
                candidates_with_max_intensity = 1;
                // logger.trace(
                //   "New max intensity found: {} at ({}, {}, {}) linear_index: {}",
                //   signal.intensity,
                //   signal.x,
                //   signal.y,
                //   signal.z.has_value() ? signal.z.value() : -1,
                //   signal.linear_index);
                continue;
            }

            // At this point, signal.intensity == max_intensity
            candidates_with_max_intensity++;

            // Guard: Skip tie-breaking if no current peak (shouldn't happen, but safety)
            if (peak_signal == nullptr) {
                continue;
            }

            // Deterministic tie-breaking using coordinate comparison
            bool should_update_tie = is_signal_preferred(signal, *peak_signal);

            // logger.trace(
            //   "Tie at intensity {}: current ({}, {}, {}) vs peak ({}, {}, {}), "
            //   "should_update: {}",
            //   signal.intensity,
            //   signal.x,
            //   signal.y,
            //   signal.z.value(),
            //   peak_signal->x,
            //   peak_signal->y,
            //   peak_signal->z.value(),
            //   should_update_tie);

            if (should_update_tie) {
                peak_signal = &signal;
            }
        }

        if (!peak_signal) {
            logger.error("Failed to find peak intensity signal");
            throw std::runtime_error("Failed to find peak intensity signal");
        }

        // logger.debug(
        //   "Selected peak signal: intensity={}, position=({}, {}, {}), linear_index={}, "
        //   "candidates_with_max_intensity={}",
        //   peak_signal->intensity,
        //   peak_signal->x,
        //   peak_signal->y,
        //   peak_signal->z.has_value() ? peak_signal->z.value() : -1,
        //   peak_signal->linear_index,
        //   candidates_with_max_intensity);

        // Get the cached or computed center of mass
        auto [com_x, com_y, com_z] = center_of_mass();
        // logger.debug("Center of mass: ({:.3f}, {:.3f}, {:.3f})", com_x, com_y, com_z);

        // Calculate the Euclidean distance
        float dx = (static_cast<float>(peak_signal->x) + 0.5f) - com_x;
        float dy = (static_cast<float>(peak_signal->y) + 0.5f) - com_y;
        float dz = (peak_signal->z.value() + 0.5f) - com_z;

        float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        // logger.debug("Peak-centroid distance: {:.3f} (dx={:.3f}, dy={:.3f}, dz={:.3f})",
        //              distance,
        //              dx,
        //              dy,
        //              dz);

        return distance;
    }

    // Getters for bounding box
    inline uint32_t get_x_min() const {
        return x_min_;
    }
    inline uint32_t get_x_max() const {
        return x_max_;
    }
    inline uint32_t get_y_min() const {
        return y_min_;
    }
    inline uint32_t get_y_max() const {
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
    uint32_t x_min_, x_max_;
    uint32_t y_min_, y_max_;
    int z_min_, z_max_;
    int num_pixels_;

    // com cache for lazy evaluation
    mutable bool com_cached_;
    mutable std::tuple<float, float, float> com_cache_;

    /**
     * @brief Determines if the first signal should be preferred over the second
     *        in case of intensity ties using coordinate-based tie-breaking.
     * 
     * @param candidate The signal to compare
     * @param current The current preferred signal
     * @return true if candidate should be preferred over current
     */
    bool is_signal_preferred(const Signal &candidate, const Signal &current) const;
};

/**
* @brief Filters reflections based on a minimum spot size and peak-centroid separation.
*
* The `min_spot_size` is the minimum number of pixels needed for a spot to pass the filter.
* The `max_peak_centroid_separation` is the maximum allow difference (in pixels) between
* the spot's centre of mass and the location of the peak intensity pixel.
*/
std::tuple<int, int> filter_reflections(std::vector<Reflection3D> &reflections,
                                        const uint min_spot_size,
                                        const float max_peak_centroid_separation);

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
                        const uint32_t width,
                        const uint32_t height,
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

    std::map<size_t, Signal> &get_signals() {
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
      const uint32_t width,
      const uint32_t height,
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
    std::map<size_t, Signal> signals;
    // Maps graph linear index -> vertex ID
    std::map<size_t, size_t> index_to_vertex;
    // Maps graph vertex ID -> linear index
    std::map<size_t, size_t> vertex_to_index;
    // 2D graph representing the connected components
    boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph;

    /**
     * @brief Builds a graph where vertices represent signal pixels and edges represent 
     * connectivity between neighboring pixels.
     * 
     * This function uses the `signals` map to find neighboring pixels efficiently.
     * The `index_to_vertex` is used to map linear indices (from `signals`) to graph vertex IDs.
     */
    void build_graph(const uint32_t width, const uint32_t height);

    /**
     * @brief Generates bounding boxes for connected components using the graph labels.
     * 
     * The `labels` vector maps each graph vertex to its connected component ID.
     * The `index_to_vertex` map is used to map linear indices to graph vertex IDs.
     */
    void generate_boxes(const uint32_t width,
                        const uint32_t height,
                        const uint32_t min_spot_size);
};

#endif  // CONNECTED_COMPONENTS_HPP