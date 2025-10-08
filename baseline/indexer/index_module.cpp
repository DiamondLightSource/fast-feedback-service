#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/eigen/dense.h>

//#include <Eigen/Dense>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <tuple>

#include "assign_indices.cc"
#include "xyz_to_rlp.cc"
#include "stills_predictor.cc"
#include <iostream>

namespace nb = nanobind;

Panel make_panel(double distance,
                 double beam_center_x,
                 double beam_center_y,
                 double pixel_size_x,
                 double pixel_size_y,
                 int image_size_x,
                 int image_size_y,
                 double thickness,
                 double mu) {
    std::array<double, 2> beam_center = {beam_center_x, beam_center_y};
    std::array<double, 2> pixel_size = {pixel_size_x, pixel_size_y};
    std::array<int, 2> image_size = {image_size_x, image_size_y};
    Panel panel(
      distance, beam_center, pixel_size, image_size, "x", "-y", thickness, mu);
    return panel;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> ssx_predict(
  std::vector<int> miller_index,
  Eigen::Vector3d s0,
  Eigen::Matrix3d UB,
  Panel panel
){
  ReflectionTable refls;
  refls.add_column("miller_index", miller_index.size() / 3, 3, miller_index);
  simple_still_reflection_predictor(s0, UB, panel, refls);
  auto s1_ = refls.column<double>("s1");
  auto &s1 = s1_.value();
  //auto xyzcal_mm_ = refls.column<double>("xyzcal.mm");
  //auto &xyzcal_mm = xyzcal_mm_.value();
  auto delpsical_ = refls.column<double>("delpsical.rad");
  auto &delpsical = delpsical_.value();
  auto xyzcal_px_ = refls.column<double>("xyzcal.px");
  auto &xyzcal_px = xyzcal_px_.value();
  // Convert mdspan to std::vector
  std::vector<double> s1_vec(s1.data_handle(), s1.data_handle() + s1.size());
  std::vector<double> xyzcal_px_vec(xyzcal_px.data_handle(), xyzcal_px.data_handle() + xyzcal_px.size());
  std::vector<double> delpsical_vec(delpsical.data_handle(), delpsical.data_handle() + delpsical.size());
  
  return std::make_tuple(std::move(s1_vec), std::move(xyzcal_px_vec), std::move(delpsical_vec));
}

std::tuple<int, std::vector<double>, Eigen::Matrix3d, Eigen::Matrix3d,
  std::vector<int>, std::vector<double>, std::vector<double>, std::vector<double>,
  std::vector<bool>, std::vector<double>>
index_from_ssx_cells(const std::vector<double> &crystal_vectors,
                     std::vector<double> rlp,
                     std::vector<double> xyzobs_px,
                     Eigen::Vector3d s0,
                     Panel panel) {
    // Note, xyzobs_mm is only used in assign_indices_global, and only the z-component
    // is used to test a condition. Given that all the z's are zero for ssx data, whether
    // in px or mm, we can safely provide xyzobs_px as input to avoid additional conversion
    // calculation.
    std::vector<double> n_indexed{};
    std::vector<double> n_unindexed{};
    std::vector<std::vector<double>> cells{};
    std::vector<Matrix3d> orientations{};
    std::vector<Matrix3d> B_matrices{};
    std::vector<std::vector<int>> miller_indices{};
    std::vector<std::vector<std::size_t>> selections{};
    mdspan_type<double> xyzobs_px_span =
      mdspan_type<double>(xyzobs_px.data(), xyzobs_px.size() / 3, 3);
    mdspan_type<double> rlp_span = mdspan_type<double>(rlp.data(), rlp.size() / 3, 3);

    for (int i = 0; i < crystal_vectors.size() / 9; ++i) {
        int start = i * 9;
        Vector3d a = {crystal_vectors[start],
                      crystal_vectors[start + 1],
                      crystal_vectors[start + 2]};
        Vector3d b = {crystal_vectors[start + 3],
                      crystal_vectors[start + 4],
                      crystal_vectors[start + 5]};
        Vector3d c = {crystal_vectors[start + 6],
                      crystal_vectors[start + 7],
                      crystal_vectors[start + 8]};
        Crystal crystal(a, b, c, *gemmi::find_spacegroup_by_name("P1"));
        // Note xyzobs not actually needed for stills assignment.
        assign_indices_results results =
          assign_indices_global(crystal.get_A_matrix(), rlp_span, xyzobs_px_span);
        n_indexed.push_back(results.number_indexed);
        n_unindexed.push_back(rlp_span.extent(0) - results.number_indexed);
        gemmi::UnitCell cell = crystal.get_unit_cell();
        std::vector<double> cell_parameters = {
          cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma};
        cells.push_back(cell_parameters);
        Matrix3d U = crystal.get_U_matrix();
        Matrix3d B = crystal.get_B_matrix();
        //std::vector<double> orientation(U.data(), U.data() + U.size());
        //std::vector<double> B_matrix(B.data(), B.data() + B.size());
        orientations.push_back(U);
        B_matrices.push_back(B);
        std::vector<int> nonzero_miller_indices;
        std::vector<std::size_t> selection;
        for (std::size_t j = 0; j < results.miller_indices.extent(0); j++) {
          const Eigen::Map<Vector3i> hkl_j(&results.miller_indices(j, 0));
          if (hkl_j[0] != 0 || hkl_j[1] != 0 || hkl_j[2] != 0){
            nonzero_miller_indices.push_back(hkl_j[0]);
            nonzero_miller_indices.push_back(hkl_j[1]);
            nonzero_miller_indices.push_back(hkl_j[2]);
            selection.push_back(j);
          }
        }
        miller_indices.push_back(nonzero_miller_indices);
        selections.push_back(selection);
    }
    auto max_iter = std::max_element(n_indexed.begin(), n_indexed.end());
    int max_index = std::distance(n_indexed.begin(), max_iter);
    std::vector<size_t> selection = selections[max_index];
    // Now take the most indexed and predict.
    ReflectionTable refls;
    refls.add_column("miller_index", miller_indices[max_index].size() / 3, 3, miller_indices[max_index]);
    Matrix3d UB = orientations[max_index] * B_matrices[max_index];
    simple_still_reflection_predictor(s0, UB, panel, refls);
    //auto s1_ = refls.column<double>("s1");
    //auto &s1 = s1_.value();
    auto delpsi_ = refls.column<double>("delpsical.rad");
    auto &delpsi = delpsi_.value();
    auto xyzcal_px_ = refls.column<double>("xyzcal.px");
    auto &xyzcal_px = xyzcal_px_.value();
    
    // now do some outlier rejection and return only the good data
    std::vector<double> xyzobs_px_indexed;
    for (auto &i : selection){
      xyzobs_px_indexed.push_back(xyzobs_px_span(i,0));
      xyzobs_px_indexed.push_back(xyzobs_px_span(i,1));
      xyzobs_px_indexed.push_back(xyzobs_px_span(i,2));
    }
    mdspan_type<double> xyzobs_px_indexed_span = mdspan_type<double>(
      xyzobs_px_indexed.data(), xyzobs_px_indexed.size() / 3, 3);


    std::vector<bool> sel(rlp.size() / 3, false);
    double rmsd_x = 0;
    double rmsd_y = 0;
    double rmsd_psi = 0;
    for (int i = 0; i < n_indexed[max_index]; i++) {
      double dx = std::pow(xyzobs_px_indexed_span(i,0) - xyzcal_px(i,0), 2);
      double dy = std::pow(xyzobs_px_indexed_span(i,1) - xyzcal_px(i,1), 2);
      double delta_r = std::sqrt(dx + dy);
      if (delta_r < 2.0){
        rmsd_x += dx;
        rmsd_y += dy;
        rmsd_psi += std::pow(delpsi(i,0),2);
        sel[selection[i]] = true;
      }
    }
    // select on the reflection table.
    ReflectionTable filtered = refls.select(sel);

    // Convert mdspan to std::vector
    delpsi_ = filtered.column<double>("delpsical.rad");
    delpsi = delpsi_.value();
    xyzcal_px_ = filtered.column<double>("xyzcal.px");
    xyzcal_px = xyzcal_px_.value();
    auto s1_ = filtered.column<double>("s1");
    auto &s1 = s1_.value();
    auto midx_ = filtered.column<int>("miller_index");
    auto &midx = midx_.value();
    //auto &xyzcal_px = xyzcal_px_.value();
    std::vector<int> miller_index_vec(midx.data_handle(), midx.data_handle() + midx.size());
    std::vector<double> s1_vec(s1.data_handle(), s1.data_handle() + s1.size());
    std::vector<double> xyzcal_px_vec(xyzcal_px.data_handle(), xyzcal_px.data_handle() + xyzcal_px.size());
    std::vector<double> delpsi_vec(delpsi.data_handle(), delpsi.data_handle() + delpsi.size());
    
    std::vector<double> rmsds;
    if (delpsi.size()){
      rmsd_x = std::sqrt(rmsd_x / delpsi.size());
      rmsd_y = std::sqrt(rmsd_y / delpsi.size());
      rmsd_psi = std::sqrt(rmsd_psi / delpsi.size());
      rmsds.push_back(rmsd_x);
      rmsds.push_back(rmsd_y);
      rmsds.push_back(rmsd_psi);
    }

    return std::make_tuple(
      n_indexed[max_index],
      cells[max_index],
      orientations[max_index],
      B_matrices[max_index],
      std::move(miller_index_vec),
      std::move(xyzcal_px_vec),
      std::move(s1_vec),
      std::move(delpsi_vec),
      std::move(sel),
      rmsds);

    /*return std::make_tuple(n_indexed[max_index],
                            cells[max_index],
                            orientations[max_index],
                            B_matrices[max_index],
                            miller_indices[max_index]);*/
}

NB_MODULE(index, m) {
    nb::class_<Panel>(m, "Panel")
      .def(nb::init<double,
                    std::array<double, 2>,
                    std::array<double, 2>,
                    std::array<int, 2>,
                    std::string,
                    std::string,
                    double,
                    double>());
    m.def("make_panel", &make_panel, "Create a configured Panel object");
    m.def("ssx_xyz_to_rlp",
          &ssx_xyz_to_rlp,
          nb::arg("xyzobs_px"),
          nb::arg("wavelength"),
          nb::arg("panel"),
          "Convert detector pixel positions to reciprocal lattice points");
    m.def("index_from_ssx_cells",
          &index_from_ssx_cells,
          nb::arg("crystal_vectors"),
          nb::arg("rlp"),
          nb::arg("xyzobs_mm"),
          nb::arg("s0"),
          nb::arg("panel"),
          "Return the maximum number of indexed reflections and cell parameters and "
          "miller indices");
    m.def("ssx_predict",
          &ssx_predict,
          nb::arg("miller_index"),
          nb::arg("s0"),
          nb::arg("UB"),
          nb::arg("panel"),
          "Predict xyzcal");
}