/**
 * @file predict.cc
 * @brief This program implements the algorithm for spot prediction.
 *
 * ReekeIndexGenerator outlined in in the LURE workshop notes.
 *
 */

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cmath>
#include <common.hpp>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>
#include <exception>
#include <fstream>
#include <gemmi/symmetry.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>
#include <tuple>

#include "index_generators.cc"
#include "ray_predictors.cc"
#include "threadpool.cc"
#include "utils.cc"

using json = nlohmann::json;

using Eigen::Matrix3d;
using Eigen::Vector3d;

constexpr size_t predicted_flag = (1 << 0);


struct predicted_data_rotation {
    // Shape {size, 3}
    std::vector<int> hkl;
    std::vector<double> s1;
    std::vector<double> xyz_px;
    std::vector<double> xyz_mm;
    // Shape {size, 1}
    std::vector<int> panels;
    std::vector<bool> enter;
    std::vector<size_t> flags;
    std::vector<int> ids;
    std::vector<uint64_t> experiment_ids;
    std::vector<std::string> identifiers;

    void add(const std::array<int, 3> &hkl_entry,
             const std::array<double, 3> &s1_entry,
             const std::array<double, 3> &xyz_px_entry,
             const std::array<double, 3> &xyz_mm_entry,
             int panel,
             bool enter_flag,
             size_t flag) {
        hkl.insert(hkl.end(), hkl_entry.begin(), hkl_entry.end());
        s1.insert(s1.end(), s1_entry.begin(), s1_entry.end());
        xyz_px.insert(xyz_px.end(), xyz_px_entry.begin(), xyz_px_entry.end());
        xyz_mm.insert(xyz_mm.end(), xyz_mm_entry.begin(), xyz_mm_entry.end());
        panels.push_back(panel);
        enter.push_back(enter_flag);
        flags.push_back(flag);
    }

    void merge(predicted_data_rotation&& other) {
        hkl.insert(hkl.end(), other.hkl.begin(), other.hkl.end());
        s1.insert(s1.end(), other.s1.begin(), other.s1.end());
        xyz_px.insert(xyz_px.end(), other.xyz_px.begin(), other.xyz_px.end());
        xyz_mm.insert(xyz_mm.end(), other.xyz_mm.begin(), other.xyz_mm.end());
        panels.insert(panels.end(), other.panels.begin(), other.panels.end());
        enter.insert(enter.end(), other.enter.begin(), other.enter.end());
        flags.insert(flags.end(), other.flags.begin(), other.flags.end());
    }

};

struct scan_varying_data {
    std::vector<Vector3d> s0_at_scan_points;
    std::vector<Matrix3d> A_at_scan_points;
    std::vector<Matrix3d> r_setting_at_scan_points;

    auto operator<=>(const scan_varying_data &) const = default;
};


predicted_data_rotation predict_single_image(const int image_index,
                          const scan_varying_data &sv_data,
                          const Vector3d s0,
                          const Matrix3d A,
                          const Goniometer &goniometer,
                          const Scan &scan,
                          const Detector &detector,
                          double param_dmin,
                          gemmi::GroupOps crystal_symmetry_operations) {
    bool use_mono = true;
    const Vector3d m2 = goniometer.get_rotation_axis();
    int z0 = scan.get_image_range()[0] - 1;
    // A Rotator object that generates rotations around axis m2
    const Rotator rotator(m2);
    const Matrix3d r_fixed = goniometer.get_sample_rotation();
    const Matrix3d r_setting = goniometer.get_setting_rotation();
    const double d_osc = scan.get_oscillation()[1];
    const double osc0 = scan.get_oscillation()[0];

    predicted_data_rotation output_data_this_image;
    Vector3d s0_1 =
      sv_data.s0_at_scan_points.empty() ? s0 : sv_data.s0_at_scan_points[image_index];
    Vector3d s0_2 = sv_data.s0_at_scan_points.empty()
                      ? s0
                      : sv_data.s0_at_scan_points[image_index + 1];
    Matrix3d A1 =
      sv_data.A_at_scan_points.empty() ? A : sv_data.A_at_scan_points[image_index];
    Matrix3d A2 =
      sv_data.A_at_scan_points.empty() ? A : sv_data.A_at_scan_points[image_index + 1];
    Matrix3d r_setting_1 = sv_data.r_setting_at_scan_points.empty()
                             ? r_setting
                             : sv_data.r_setting_at_scan_points[image_index];
    Matrix3d r_setting_2 = sv_data.r_setting_at_scan_points.empty()
                             ? r_setting
                             : sv_data.r_setting_at_scan_points[image_index + 1];
    // Redefine A1 and A2 to encompass all 3 rotations
    const double phi_beg = osc0 + image_index * d_osc;
    const double phi_end = phi_beg + d_osc;
    Matrix3d r_beg = rotator.rotation_matrix(phi_beg);
    Matrix3d r_end = rotator.rotation_matrix(phi_end);
    A1 = r_setting_1 * r_beg * r_fixed * A1;
    A2 = r_setting_2 * r_end * r_fixed * A2;
    ReekeIndexGenerator index_generator(
      A1, A2, crystal_symmetry_operations, s0_1, s0_2, param_dmin, use_mono);

    std::function<std::array<std::optional<Ray>, 2>(const std::array<int, 3> &)>
      predict_ray;

    // Note that using predict_ray_monochromatic_sv seems to be 2x faster than using
    // predict_ray_monochromatic_static, and we have all the inputs required, so use that...
    // Otherwise wrap in an if/else based on if any sv_data not being empty.
    predict_ray = [=](const std::array<int, 3> &index) {
        std::array<std::optional<Ray>, 2> rays;
        rays[0] = predict_ray_monochromatic_sv(
          index, A1, A2, s0_1, s0_2, param_dmin, phi_beg, d_osc);
        return rays;
    };
    // Slower static predict function,
    /*predict_ray = [=](const std::array<int, 3>& index) {
      return predict_ray_monochromatic_static(
        index, A1, r_setting_1, r_setting_1_inv, s0, m2, rotator, param_dmin, phi_beg, d_osc
      );
  };*/

    for (;;) {
        std::optional<std::array<int, 3>> index = index_generator.next();
        if (!index) break;

        // Check if a reflection occurs at the given Miller index
        // within the required resolution.
        std::array<std::optional<Ray>, 2> rays = predict_ray(index.value());
        for (std::optional<Ray> ray : rays) {
            if (!ray) continue;
            // Append the ray
            auto impact = detector.get_ray_intersection(ray->s1);
            if (!impact.has_value()) continue;
            // Get the frame that a reflection with this angle will be observed at
            double frame = z0 + (ray->angle - osc0) / d_osc;
            intersection result = impact.value();
            auto panel = result.panel_id;
            std::array<double, 3> coords_mm = {
              result.xymm[0], result.xymm[1], ray->angle * M_PI / 180};
            std::array<double, 2> xycoords_px =
              detector.panels()[panel].mm_to_px(coords_mm[0], coords_mm[1]);
            std::array<double, 3> coords_px = {xycoords_px[0], xycoords_px[1], frame};
            std::array<double, 3> s1 = {ray->s1[0], ray->s1[1], ray->s1[2]};
            output_data_this_image.add(index.value(),
                                       s1,
                                       coords_px,
                                       coords_mm,
                                       panel,
                                       ray->entering,
                                       predicted_flag);
        }
    }
    return output_data_this_image;
}

predicted_data_rotation predict_rotation(Experiment<MonochromaticBeam> &experiment,
                      const scan_varying_data &sv_data,
                      double param_dmin,
                      int buffer_size = 0,
                      int nthreads = 1) {
    Scan &scan = experiment.scan();
    if (buffer_size > 0) {
        // Can't have both scan varying data and a buffer
        // (as the models will not be defined outside the scan range)
        if (sv_data != scan_varying_data{}) {
            throw std::runtime_error(
              "Can't call predict function with scan varying data and an image "
              "buffer.");
        }
        std::array<int, 2> image_range = {scan.get_image_range()[0] - buffer_size,
                                          scan.get_image_range()[1] + buffer_size};
        std::array<double, 2> oscillation = {
          scan.get_oscillation()[0] - buffer_size * scan.get_oscillation()[1],
          scan.get_oscillation()[1]};
        scan = Scan(image_range, oscillation);
    }
    MonochromaticBeam &beam = experiment.beam();
    const Goniometer &goniometer = experiment.goniometer();
    Detector &detector = experiment.detector();
    const Crystal &crystal = experiment.crystal();

    gemmi::GroupOps crystal_symmetry_operations =
      crystal.get_space_group().operations();
    const Matrix3d A = crystal.get_A_matrix();
    const Vector3d m2 = goniometer.get_rotation_axis();
    // A Rotator object that generates rotations around axis m2
    const Rotator rotator(m2);
    const Matrix3d r_fixed = goniometer.get_sample_rotation();
    const Matrix3d r_setting = goniometer.get_setting_rotation();
    const double d_osc = scan.get_oscillation()[1];
    const double osc0 = scan.get_oscillation()[0];

    int z0 = scan.get_image_range()[0] - 1;
    int z1 = scan.get_image_range()[1];

    Vector3d s0 = beam.get_s0();
    int n_images = z1 - z0;
    nthreads = std::min(n_images, nthreads);
    logger.info("Predicting spots on {} images over {} threads", n_images, nthreads);

    std::vector<predicted_data_rotation> per_image_results(n_images);

    std::exception_ptr eptr = nullptr;
    std::mutex eptr_mtx;

    {
    ThreadPool pool(nthreads);

    for (int image_index = 0; image_index < n_images; ++image_index) {
        pool.enqueue([&, image_index] {
            try {
              per_image_results[image_index] = predict_single_image(image_index,
                                sv_data,
                                s0,
                                A,
                                goniometer,
                                scan,
                                detector,
                                param_dmin,
                                crystal_symmetry_operations);
            } catch (...){
                std::lock_guard<std::mutex> lk(eptr_mtx);
                if (!eptr) eptr = std::current_exception();
            }
        });
    }
    } // Threadpool needs to go out of scope to make sure all
    // jobs completed before merge
    if (eptr) std::rethrow_exception(eptr);
    
    predicted_data_rotation all_output;
    for (auto& r : per_image_results) {
        all_output.merge(std::move(r));
    }
    return all_output;
}

std::tuple<bool, scan_varying_data> extract_scan_varying_data(json elist_json_obj, Scan scan){
  scan_varying_data sv_data;
  bool scan_varying = false;

  const int num_images =
    scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
  json beam_data = elist_json_obj.at("beam")[0];
  json goniometer_data = elist_json_obj.at("goniometer")[0];
  json crystal_data = elist_json_obj.at("crystal")[0];
  json s0_at_scan_points;
  json A_at_scan_points;
  json r_setting_at_scan_points;
  if (beam_data.contains("s0_at_scan_points")) {
      scan_varying = true;
      s0_at_scan_points = beam_data.at("s0_at_scan_points");
      if (s0_at_scan_points.size()
          == num_images + 1) {  // i.e. is expected length.
          std::vector<Vector3d> scan_varying_s0;
          for (const auto &entry : s0_at_scan_points) {
              Vector3d vec(entry[0].get<double>(),
                            entry[1].get<double>(),
                            entry[2].get<double>());
              scan_varying_s0.push_back(vec);
          }
          sv_data.s0_at_scan_points = scan_varying_s0;
      }
  }
  if (crystal_data.contains("A_at_scan_points")) {
      scan_varying = true;
      A_at_scan_points = crystal_data.at("A_at_scan_points");
      if (A_at_scan_points.size() == num_images + 1) {
          std::vector<Matrix3d> scan_varying_A;
          for (const auto &entry : A_at_scan_points) {
              Matrix3d A_mat;
              A_mat << entry[0].get<double>(), entry[1].get<double>(),
                entry[2].get<double>(), entry[3].get<double>(),
                entry[4].get<double>(), entry[5].get<double>(),
                entry[6].get<double>(), entry[7].get<double>(),
                entry[8].get<double>();
              scan_varying_A.push_back(A_mat);
          }
          sv_data.A_at_scan_points = scan_varying_A;
      }
  }
  if (goniometer_data.contains("setting_rotation_at_scan_points")) {
      scan_varying = true;
      r_setting_at_scan_points =
        goniometer_data.at("setting_rotation_at_scan_points");
      if (r_setting_at_scan_points.size() == num_images + 1) {
          std::vector<Matrix3d> scan_varying_r;
          for (const auto &entry : r_setting_at_scan_points) {
              Matrix3d r_mat;
              r_mat << entry[0].get<double>(), entry[1].get<double>(),
                entry[2].get<double>(), entry[3].get<double>(),
                entry[4].get<double>(), entry[5].get<double>(),
                entry[6].get<double>(), entry[7].get<double>(),
                entry[8].get<double>();
              scan_varying_r.push_back(r_mat);
          }
          sv_data.r_setting_at_scan_points = scan_varying_r;
      }
  }
  return std::make_tuple(scan_varying, sv_data);
}