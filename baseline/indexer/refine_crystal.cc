#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <vector>

#include "assign_indices.cc"
#include "ffs_logger.hpp"
#include "refine_candidate.cc"
#include "reflection_filter.cc"

void refine_crystal(Crystal &crystal,
                    ReflectionTable const &obs,
                    Goniometer gonio,
                    MonochromaticBeam &beam,
                    Panel &panel,
                    double scan_width) {
    std::vector<int> miller_indices_data;
    int count;
    auto preassign = std::chrono::system_clock::now();

    // First assign miller indices to the data using the crystal model.
    auto rlp_ = obs.column<double>("rlp");
    const mdspan_type<double> &rlp = rlp_.value();
    auto xyzobs_mm_ = obs.column<double>("xyzobs.mm.value");
    const mdspan_type<double> &xyzobs_mm = xyzobs_mm_.value();
    assign_indices_results results =
      assign_indices_global(crystal.get_A_matrix(), rlp, xyzobs_mm);
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - preassign;
    logger.debug("Time for assigning indices: {:.5f}s", elapsed_time.count());

    logger.info("Indexed {}/{} reflections",
                results.number_indexed,
                results.miller_indices.extent(0));

    auto t3 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time1 = t3 - t2;
    logger.debug("Time for correct: {:.5f}s", elapsed_time1.count());

    // Perform filtering of the data prior to candidate refinement.
    ReflectionTable sel_obs = reflection_filter_preevaluation(
      obs, results.miller_indices, gonio, crystal, beam, panel, scan_width, 100);
    auto postfilter = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_timefilter = postfilter - t3;
    logger.debug("Time for reflection_filter: {:.5f}s", elapsed_timefilter.count());

    auto t4 = std::chrono::system_clock::now();
    double xyrmsd = refine_indexing_candidate(crystal, gonio, beam, panel, sel_obs);
    std::chrono::duration<double> elapsed_time_refine =
      std::chrono::system_clock::now() - t4;
    auto flags_ = sel_obs.column<std::size_t>("flags");
    int n_refl = flags_.value().extent(0);
    logger.debug("Time for refinement: {:.5f}s", elapsed_time_refine.count());
    logger.info("rmsd_xy {:.5f} on {} reflections", xyrmsd, n_refl);
}