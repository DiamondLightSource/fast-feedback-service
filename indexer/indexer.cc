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

using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::Vector3i;
using json = nlohmann::json;

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
    auto args = parser.parse_args(argc, argv);

    if (!parser.is_used("--expt")){
        fmt::print("Error: must specify experiment list file with --expt\n");
        std::exit(1);
    }
    if (!parser.is_used("--refl")){
        fmt::print("Error: must specify spotfinding results file (in DIALS HDF5 format) with --refl\n");
        std::exit(1);
    }
    std::string imported_expt = parser.get<std::string>("--expt");
    std::string filename = parser.get<std::string>("--refl");
    int max_refine = parser.get<int>("max-refine");
    
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
    json beam_data = elist_json_obj["beam"][0];
    MonoXrayBeam beam(beam_data);
    /*json beam_out = beam.to_json();
    std::ofstream file("test_beam.json");
    file << beam_out.dump(4);*/
    json scan_data = elist_json_obj["scan"][0];
    Scan scan(scan_data);
    /*json scan_out = scan.to_json();
    std::ofstream scanfile("test_scan.json");
    scanfile << scan_out.dump(4);*/

    json gonio_data = elist_json_obj["goniometer"][0];
    Goniometer gonio(gonio_data);
    /*json gonio_out = gonio.to_json();
    std::ofstream goniofile("test_gonio.json");
    goniofile << gonio_out.dump(4);*/

    json panel_data = elist_json_obj["detector"][0]["panels"][0];
    Panel detector(panel_data);
    json det_out = detector.to_json();
    std::ofstream detfile("test_detector.json");
    detfile << det_out.dump(4);

    //TODO
    // implement max cell/d_min estimation. - will need annlib if want same result as dials.

    // get processed reflection data from spotfinding
    //std::string filename = "/dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24_rot/strong.refl";
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    std::vector<double> data = read_xyzobs_data(filename, array_name);

    std::vector<Vector3d> rlp = xyz_to_rlp(data, detector, beam, scan, gonio);

    // compare against dials proc
    /*std::string filename2 = "/dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24_rot/index_rot.refl";
    std::string array_name2 = "/dials/processing/group_0/rlp";
    std::vector<double> data2 = read_xyzobs_data(filename2, array_name2);
    for (int i=0;i<rlp.size();i++){
        std::cout<< i << " " << rlp[i][2] << " " << data2[i*3+2] << std::endl;
        //assert(rlp[i] == data2[i*3+2]);
    }*/
    //assert(false);

    std::cout << "Number of reflections: " << rlp.size() << std::endl;
    
    double d_min = 1.31;
    double b_iso = -4.0 * std::pow(d_min, 2) * log(0.05);
    double max_cell = 33.8;
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

    CandidateOrientationMatrices candidates(candidate_vecs, 1000);
    std::vector<Vector3i> miller_indices;
    // first extract phis
    int image_range_start = scan.get_image_range()[0];
    std::vector<double> phi(rlp.size());
    std::array<double, 2> oscillation = scan.get_oscillation();
    double osc_width = oscillation[1];
    double osc_start = oscillation[0];
    double DEG2RAD = M_PI / 180.0;
    for (int i=0;i<phi.size();i++){
        phi[i] = (((data[i*3+2] +1 - image_range_start)*osc_width) + osc_start) * DEG2RAD;
    }
    Vector3i null{{0,0,0}};
    int n = 0;
    while (candidates.has_next() && n < max_refine){
        Crystal crystal = candidates.next();
        std::cout << crystal.get_A_matrix() << std::endl;
        n++;
        
        std::vector<Vector3i> miller_indices = assign_indices_global(crystal.get_A_matrix(), rlp, phi);
        int count = 0;
        for (int i=0;i<miller_indices.size();i++){
            if (miller_indices[i] != null){
                count++;
                //std::cout << i << ": " << miller_indices[i][0] << " " << miller_indices[i][1] << " " << miller_indices[i][2] << std::endl;
            }
        }
        std::cout << count << " nonzero miller indices" << std::endl;
        // now need to select on dmin (and first rot of scan)
        // then index_reflections (i.e. assign indices) - global method with hkl tolerance of 0.3
        // requires data items phi (xyzobs.mm.value[2]), crystal A matrix, tol, 
    }
    
    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    std::cout << "Total time for indexer: " << elapsed_time.count() << "s" << std::endl;

}