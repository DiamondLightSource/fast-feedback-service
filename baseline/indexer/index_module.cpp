#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <tuple>

#include "assign_indices.cc"
#include "stills_predictor.cc"
#include "xyz_to_rlp.cc"

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

struct IndexingResult {
    std::vector<double> cell_parameters;
    Eigen::Matrix3d A_matrix;
    std::vector<int> miller_indices;
    std::vector<double> xyzobs_px;
    std::vector<double> xyzcal_px;
    std::vector<double> s1;
    std::vector<double> delpsi;
    std::vector<double> rmsds;
};

IndexingResult index_from_ssx_cells(const std::vector<double> &crystal_vectors,
                                    std::vector<double> rlp_data,
                                    std::vector<double> xyzobs_px_data,
                                    Eigen::Vector3d s0,
                                    Panel panel) {
    // Convert the raw input data arrays to spans
    mdspan_type<double> xyzobs_px =
      mdspan_type<double>(xyzobs_px_data.data(), xyzobs_px_data.size() / 3, 3);
    mdspan_type<double> rlp =
      mdspan_type<double>(rlp_data.data(), rlp_data.size() / 3, 3);

    // first convert the cells vector to crystals
    std::vector<Crystal> crystals{};
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
        crystals.push_back(crystal);
    }

    // Choose the best crystal based on number of indexed.
    std::vector<double> n_indexed{};
    std::vector<std::vector<double>> cells{};
    std::vector<Matrix3d> A_matrices{};
    std::vector<std::vector<int>> miller_indices{};
    std::vector<std::vector<std::size_t>> selections{};

    for (auto &crystal : crystals) {
        // Note xyzobs not actually needed for stills assignment.
        assign_indices_results results =
          assign_indices_global(crystal.get_A_matrix(), rlp, xyzobs_px);
        n_indexed.push_back(results.number_indexed);
        gemmi::UnitCell cell = crystal.get_unit_cell();
        std::vector<double> cell_parameters = {
          cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma};
        cells.push_back(cell_parameters);
        Matrix3d U = crystal.get_U_matrix();
        Matrix3d B = crystal.get_B_matrix();
        A_matrices.push_back(U * B);
        std::vector<int> nonzero_miller_indices;
        std::vector<std::size_t> selection;
        for (std::size_t j = 0; j < results.miller_indices.extent(0); j++) {
            const Eigen::Map<Vector3i> hkl_j(&results.miller_indices(j, 0));
            if ((hkl_j[0] != 0) || (hkl_j[1] != 0) || (hkl_j[2] != 0)) {
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
    std::vector<int> best_miller_indices = miller_indices[max_index];
    int best_n_indexed = n_indexed[max_index];
    Matrix3d A = A_matrices[max_index];

    // Select on the input xyzobs_px to get the values for only indexed reflections
    std::vector<double> xyzobs_px_indexed_data;
    for (auto &i : selection) {
        xyzobs_px_indexed_data.push_back(xyzobs_px(i, 0));
        xyzobs_px_indexed_data.push_back(xyzobs_px(i, 1));
        xyzobs_px_indexed_data.push_back(xyzobs_px(i, 2));
    }
    mdspan_type<double> xyzobs_px_indexed = mdspan_type<double>(
      xyzobs_px_indexed_data.data(), xyzobs_px_indexed_data.size() / 3, 3);

    // Now predict (generates xyzcal.px and delpsi).
    ReflectionTable refls;
    refls.add_column(
      "miller_index", best_miller_indices.size() / 3, 3, best_miller_indices);
    simple_still_reflection_predictor(s0, A, panel, refls);

    // now do some outlier rejection and return only the good data
    auto delpsi_ = refls.column<double>("delpsical.rad");
    auto &delpsi = delpsi_.value();
    auto xyzcal_px_ = refls.column<double>("xyzcal.px");
    auto &xyzcal_px = xyzcal_px_.value();

    std::vector<bool> sel_for_indexed(best_n_indexed, false);
    double rmsd_x = 0;
    double rmsd_y = 0;
    double rmsd_psi = 0;
    int n = 0;
    for (std::size_t i = 0; i < best_n_indexed; i++) {
        double dx = std::pow(xyzobs_px_indexed(i, 0) - xyzcal_px(i, 0), 2);
        double dy = std::pow(xyzobs_px_indexed(i, 1) - xyzcal_px(i, 1), 2);
        double delta_r = std::sqrt(dx + dy);
        if (delta_r < 2.0) {  // crude filter for now.
            rmsd_x += dx;
            rmsd_y += dy;
            rmsd_psi += std::pow(delpsi(i, 0), 2);
            sel_for_indexed[i] = true;
            n += 1;
        }
    }
    std::vector<double> rmsds;
    if (n > 0) {
        rmsd_x = std::sqrt(rmsd_x / n);
        rmsd_y = std::sqrt(rmsd_y / n);
        rmsd_psi = std::sqrt(rmsd_psi / n);
        rmsds.push_back(rmsd_x);
        rmsds.push_back(rmsd_y);
        rmsds.push_back(rmsd_psi);
    }

    // select on the reflection table.
    refls.add_column(
      "xyzobs.px.value", xyzobs_px_indexed.extent(0), 3, xyzobs_px_indexed_data);
    ReflectionTable filtered = refls.select(sel_for_indexed);

    // Get the data arrays to return.
    delpsi_ = filtered.column<double>("delpsical.rad");
    delpsi = delpsi_.value();
    xyzcal_px_ = filtered.column<double>("xyzcal.px");
    xyzcal_px = xyzcal_px_.value();
    auto xyzobs_px_filtered_ = filtered.column<double>("xyzobs.px.value");
    auto &xyzobs_px_filtered = xyzobs_px_filtered_.value();
    auto s1_ = filtered.column<double>("s1");
    auto &s1 = s1_.value();
    auto midx_ = filtered.column<int>("miller_index");
    auto &midx = midx_.value();
    std::vector<int> miller_index_vec(midx.data_handle(),
                                      midx.data_handle() + midx.size());
    std::vector<double> s1_vec(s1.data_handle(), s1.data_handle() + s1.size());
    std::vector<double> xyzcal_px_vec(xyzcal_px.data_handle(),
                                      xyzcal_px.data_handle() + xyzcal_px.size());
    std::vector<double> xyzobs_px_vec(
      xyzobs_px_filtered.data_handle(),
      xyzobs_px_filtered.data_handle() + xyzobs_px_filtered.size());
    std::vector<double> delpsi_vec(delpsi.data_handle(),
                                   delpsi.data_handle() + delpsi.size());

    return IndexingResult(cells[max_index],
                          A,
                          std::move(miller_index_vec),
                          std::move(xyzobs_px_vec),
                          std::move(xyzcal_px_vec),
                          std::move(s1_vec),
                          std::move(delpsi_vec),
                          rmsds);
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
    nb::class_<IndexingResult>(m, "IndexingResult")
      .def_prop_ro("cell_parameters",
                   [](const IndexingResult &r) { return r.cell_parameters; })
      .def_prop_ro("A_matrix", [](const IndexingResult &r) { return r.A_matrix; })
      .def_prop_ro("miller_indices",
                   [](const IndexingResult &r) { return r.miller_indices; })
      .def_prop_ro("xyzobs_px", [](const IndexingResult &r) { return r.xyzobs_px; })
      .def_prop_ro("xyzcal_px", [](const IndexingResult &r) { return r.xyzcal_px; })
      .def_prop_ro("s1", [](const IndexingResult &r) { return r.s1; })
      .def_prop_ro("delpsi", [](const IndexingResult &r) { return r.delpsi; })
      .def_prop_ro("rmsds", [](const IndexingResult &r) { return r.rmsds; });
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
          nb::arg("xyzobs_px"),
          nb::arg("s0"),
          nb::arg("panel"),
          "Assign miller indices to the best crystal, predict reflections and "
          "calculate rnmsds");
}