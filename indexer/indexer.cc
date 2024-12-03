#include <iostream>
#include <lodepng.h>
#include <string>
#include <nlohmann/json.hpp>
#include "common.hpp"
#include "cuda_common.hpp"
#include "h5read.h"
#include <vector>
#include <cstring>
#include <Eigen/Dense>
#include "xyz_to_rlp.cc"
#include "flood_fill.cc"
#include "sites_to_vecs.cc"
#include "fft3d.cc"
#include "assign_indices.h"
#include "reflection_data.h"
#include "scanstaticpredictor.cc"
#include "combinations.cc"
#include <chrono>
#include <fstream>
#include <dx2/detector.h>
#include <dx2/beam.h>
#include <dx2/scan.h>
#include <dx2/crystal.h>
#include <dx2/goniometer.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/os.h>
#include "refman_filter.cc"

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::Vector3i;
using json = nlohmann::json;

struct score_and_crystal {
    double score;
    Crystal crystal;
    double num_indexed;
    double rmsdxy;
};

int main(int argc, char **argv) {

    auto t1 = std::chrono::system_clock::now();
    auto parser = CUDAArgumentParser();
    parser.add_argument("-e", "--expt")
      .help("Path to the DIALS expt file");
    parser.add_argument("-r", "--refl")
      .help("Path to the h5 reflection table file containing spotfinding results");
    parser.add_argument("--max-refine")
      .help("Maximum number of crystal models to test")
      .default_value<int>(50)
      .scan<'i', int>();
    parser.add_argument("--dmin")
      .help("Resolution limit")
      .default_value<float>(1.0)
      .scan<'f', float>();
    parser.add_argument("--max-cell")
      .help("The maxiumu cell length to try during indexing")
      .scan<'f', float>();
    auto args = parser.parse_args(argc, argv);

    if (!parser.is_used("--expt")){
        fmt::print("Error: must specify experiment list file with --expt\n");
        std::exit(1);
    }
    if (!parser.is_used("--refl")){
        fmt::print("Error: must specify spotfinding results file (in DIALS HDF5 format) with --refl\n");
        std::exit(1);
    }
    if (!parser.is_used("--max-cell")){
        fmt::print("Error: must specify --max-cell\n");
        std::exit(1);
    }
    std::string imported_expt = parser.get<std::string>("--expt");
    std::string filename = parser.get<std::string>("--refl");
    int max_refine = parser.get<int>("max-refine");
    double max_cell = parser.get<float>("max-cell");
    double d_min = parser.get<float>("dmin");
    
    //std::string imported_expt = "/dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24_rot/imported.expt";
    std::ifstream f(imported_expt);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    }
    catch(json::parse_error& ex){
        std::cerr << "Error: Unable to read " << imported_expt.c_str() << "; json parse error at byte " << ex.byte << std::endl;
        std::exit(1);
    }

    // Load the models
    json beam_data = elist_json_obj["beam"][0];
    MonoXrayBeam beam(beam_data);
    json scan_data = elist_json_obj["scan"][0];
    Scan scan(scan_data);
    json gonio_data = elist_json_obj["goniometer"][0];
    Goniometer gonio(gonio_data);
    json panel_data = elist_json_obj["detector"][0]["panels"][0];
    Panel detector(panel_data);

    //TODO
    // implement max cell/d_min estimation. - will need annlib if want same result as dials.

    // get processed reflection data from spotfinding
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    std::vector<double> xyzobs_px = read_array_from_h5_file<double>(filename, array_name);
    //read_xyzobs_data(filename, array_name);
    std::string flags_array_name = "/dials/processing/group_0/flags";
    std::vector<std::size_t> flags = read_array_from_h5_file<std::size_t>(filename, flags_array_name);
    //read_flags_data(filename, flags_array_name);

    std::vector<Vector3d> rlp = xyz_to_rlp(xyzobs_px, detector, beam, scan, gonio);

    // some more setup - entering flags and s1
    //self.reflections.centroid_px_to_mm(self.experiments)
    //self.reflections.map_centroids_to_reciprocal_space(self.experiments)
    //self.reflections.calculate_entering_flags(self.experiments)
    Vector3d s0 = beam.get_s0();
    Vector3d axis = gonio.get_rotation_axis();
    Matrix3d d_matrix = detector.get_d_matrix();
    std::array<double, 2> oscillation = scan.get_oscillation();
    double osc_width = oscillation[1];
    double osc_start = oscillation[0];
    int image_range_start = scan.get_image_range()[0];
    double DEG2RAD = M_PI / 180.0;

    // calculate s1 and xyzobsmm
    std::vector<Vector3d> s1(rlp.size());
    std::vector<Vector3d> xyzobs_mm(rlp.size());
    std::vector<Vector3d> xyzcal_mm(rlp.size());
    std::vector<double> phi(rlp.size());
    for (int i = 0; i < rlp.size(); ++i) {
        int vec_idx= 3*i;
        double x1 = xyzobs_px[vec_idx];
        double x2 = xyzobs_px[vec_idx+1];
        double x3 = xyzobs_px[vec_idx+2];
        std::array<double, 2> xymm = detector.px_to_mm(x1,x2);
        double rot_angle = (((x3 + 1 - image_range_start) * osc_width) + osc_start) * DEG2RAD;
        phi[i] = rot_angle;
        Vector3d m = {xymm[0], xymm[1], 1.0};
        s1[i] = d_matrix * m;
        xyzobs_mm[i] = {xymm[0], xymm[1], rot_angle};
    }
    
    // calculate entering array
    std::vector<bool> enterings(rlp.size());
    Vector3d vec = s0.cross(axis);
    for (int i=0;i<s1.size();i++){
        enterings[i] = ((s1[i].dot(vec)) < 0.0);
    }

    std::cout << "Number of reflections: " << rlp.size() << std::endl;
    
    //double d_min = 1.31;
    //double d_min = 1.84;
    double b_iso = -4.0 * std::pow(d_min, 2) * log(0.05);
    //double max_cell = 33.8;
    //double max_cell = 94.4;
    std::cout << "Setting b_iso =" << b_iso << std::endl;
    
    std::vector<double> real_fft(256*256*256, 0.0);
    std::vector<bool> used_in_indexing = fft3d(rlp, real_fft, d_min, b_iso);
    
    std::cout << real_fft[0] << " " << used_in_indexing[0] << std::endl;

    std::vector<int> grid_points_per_void;
    std::vector<Vector3d> centres_of_mass_frac;
    std::tie(grid_points_per_void, centres_of_mass_frac) = flood_fill(real_fft);
    std::tie(grid_points_per_void, centres_of_mass_frac) = flood_fill_filter(grid_points_per_void, centres_of_mass_frac, 0.15);
    
    std::vector<Vector3d> candidate_vecs =
        sites_to_vecs(centres_of_mass_frac, grid_points_per_void, d_min, 3.0, max_cell);

    std::string n_vecs = std::to_string(candidate_vecs.size() - 1);
    size_t n_zero = n_vecs.length();
    json vecs_out;
    for (int i=0;i<candidate_vecs.size();i++){
        std::string s = std::to_string(i);
        auto pad_s = std::string(n_zero - std::min(n_zero, s.length()), '0') + s;
        vecs_out[pad_s] = candidate_vecs[i];
    }
    std::ofstream vecs_file("candidate_vectors.json");
    vecs_file << vecs_out.dump(4);

    CandidateOrientationMatrices candidates(candidate_vecs, 1000);
    std::vector<Vector3i> miller_indices;
    // first extract phis
    // need to select on dmin, also only first 360 deg of scan. Do this earlier?
    /*int image_range_start = scan.get_image_range()[0];
    std::vector<double> phi_select(rlp.size());
    std::vector<Vector3d> rlp_select(rlp.size());
    std::array<double, 2> oscillation = scan.get_oscillation();
    double osc_width = oscillation[1];
    double osc_start = oscillation[0];
    double DEG2RAD = M_PI / 180.0;*/

    // Fix this inefficient selection with reflection-table-like struct.
    std::vector<double> phi_select(rlp.size());
    std::vector<Vector3d> rlp_select(rlp.size());
    std::vector<std::size_t> flags_select(rlp.size());
    std::vector<Vector3d> xyzobs_mm_select(rlp.size());
    std::vector<Vector3d> xyzcal_mm_select(rlp.size());
    std::vector<Vector3d> s1_select(rlp.size());
    std::vector<bool> entering_select(rlp.size());
    int selcount=0;
    // also select flags, xyzobs/cal, s1 and enterings
    for (int i=0;i<phi_select.size();i++){
        if ((1.0/rlp[i].norm()) > d_min){
            phi_select[selcount] = phi[i];
            rlp_select[selcount] = rlp[i];
            flags_select[selcount] = flags[i];
            xyzobs_mm_select[selcount] = xyzobs_mm[i];
            xyzcal_mm_select[selcount] = xyzcal_mm[i];
            s1_select[selcount] = s1[i];
            entering_select[selcount] = enterings[i];
            selcount++;
        }
    }
    rlp_select.resize(selcount);
    phi_select.resize(selcount);
    flags_select.resize(selcount);
    xyzobs_mm_select.resize(selcount);
    xyzcal_mm_select.resize(selcount);
    s1_select.resize(selcount);
    entering_select.resize(selcount);
    Vector3i null{{0,0,0}};
    int n = 0;

    // iterate over candidates; assign indices, refine, score.
    // need a map of scores for candidates: index to score and xtal. What about miller indices?

    std::map<int,score_and_crystal> results_map;

    while (candidates.has_next() && n < max_refine){
        Crystal crystal = candidates.next();
        n++;
        std::vector<Vector3i> miller_indices = assign_indices_global(crystal.get_A_matrix(), rlp_select, phi_select);
        
        // for debugging, let's count the number of nonzero miller indices
        int count = 0;
        for (int i=0;i<miller_indices.size();i++){
            if (miller_indices[i] != null){
                count++;
                //std::cout << i << ": " << miller_indices[i][0] << " " << miller_indices[i][1] << " " << miller_indices[i][2] << std::endl;
            }
        }
        //std::cout << count << " nonzero miller indices" << std::endl;

        // make a reflection table like object
        reflection_data obs;
        obs.miller_indices = miller_indices;
        obs.flags = flags_select;
        obs.xyzobs_mm = xyzobs_mm_select;
        obs.xyzcal_mm = xyzcal_mm_select;
        obs.s1 = s1_select;
        obs.entering = entering_select;

        int n_images = scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
        double width = scan.get_oscillation()[0] + (scan.get_oscillation()[1] * n_images);

        // get a filtered selection for refinement
        reflection_data sel_obs = reflection_filter_preevaluation(
            obs, gonio, crystal, beam, detector, width, 20
        );

        // do some refinement

        // now calculate the rmsd and model likelihood
        double xsum = 0;
        double ysum = 0;
        double zsum = 0;
        for (int i=0;i<sel_obs.flags.size();i++){
            /*if (sel_obs.miller_indices[i] == null){
                continue;
            }*/
            Vector3d xyzobs = sel_obs.xyzobs_mm[i];
            Vector3d xyzcal = sel_obs.xyzcal_mm[i];
            xsum += std::pow(xyzobs[0] - xyzcal[0],2);
            ysum += std::pow(xyzobs[1] - xyzcal[1],2);
            zsum += std::pow(xyzobs[2] - xyzcal[2],2);
        }
        double rmsdx = std::pow(xsum / sel_obs.xyzcal_mm.size(), 0.5);
        double rmsdy = std::pow(ysum / sel_obs.xyzcal_mm.size(), 0.5);
        double rmsdz = std::pow(zsum / sel_obs.xyzcal_mm.size(), 0.5);
        double xyrmsd = std::pow(std::pow(rmsdx, 2)+std::pow(rmsdy, 2), 0.5);
        //std::cout << "RMSDx: " << rmsdx << ", RMSDy: " << rmsdy << ", RMSDz: " << rmsdz << std::endl;
        //std::cout << xyrmsd << " <<xyrmsd" << std::endl;
        //s1, flags, xyzcal.mm*/

        // FIXME refine the crystal model
        // skip from dials.algorithms.indexing import non_primitive_basis step
        //
        // data needed: flags, s1, xyzobs.mm.value, entering.
        // call 'calculate_entering_flags'
        // flags from file
        // s1 calculated
        // xyzobs.mm.value already effectively calculated in xyz_to_rlp?

        // FIXME score the refined model

        score_and_crystal sac;
        sac.score = (double)n;
        sac.crystal = crystal;
        sac.num_indexed = count;
        sac.rmsdxy = xyrmsd;
        results_map[n] = sac;
        
    }
    std::cout << "Unit cell, #indexed, rmsd_xy" << std::endl;
    for (auto it=results_map.begin();it!=results_map.end();it++){
        gemmi::UnitCell cell = (*it).second.crystal.get_unit_cell();
        std::string printcell;
        std::string a = std::to_string(cell.a);
        std::string b = std::to_string(cell.b);
        std::string c = std::to_string(cell.c);
        std::string al = std::to_string(cell.alpha);
        std::string be = std::to_string(cell.beta);
        std::string ga = std::to_string(cell.gamma);
        std::cout << a << ", " << b << ", " << c << ", " << al << ", " << be << ", "<< ga << ", "<< (*it).second.num_indexed << ", " << (*it).second.rmsdxy << std::endl;
    }

    // find the best crystal from the map - lowest score
    auto it = *std::min_element(results_map.begin(), results_map.end(),
            [](const auto& l, const auto& r) { return l.second.score < r.second.score; });
    Crystal best_xtal = it.second.crystal;
    // save the best crystal.
    json cryst_out = best_xtal.to_json();
    std::ofstream cfile("best_crystal.json");
    cfile << cryst_out.dump(4);

    json elist_out;
    elist_out["__id__"] = "ExperimentList";
    json expt_out;
    // no imageset for now.
    expt_out["__id__"] = "Experiment";
    expt_out["identifier"] = "test";
    expt_out["beam"] = 0;
    expt_out["detector"] = 0;
    expt_out["goniometer"] = 0;
    expt_out["scan"] = 0;
    expt_out["crystal"] = 0;
    elist_out["experiment"] = std::array<json, 1> {expt_out};
    elist_out["crystal"] = std::array<json, 1> {cryst_out};
    elist_out["scan"] = std::array<json, 1> {scan.to_json()};
    elist_out["goniometer"] = std::array<json, 1> {gonio.to_json()};
    elist_out["beam"] = std::array<json, 1> {beam.to_json()};
    elist_out["detector"] = std::array<json, 1> {detector.to_json()};

    std::ofstream efile("elist.json");
    efile << elist_out.dump(4);

    
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    std::cout << "Total time for indexer: " << elapsed_time.count() << "s" << std::endl;

}