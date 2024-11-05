#include <iostream>
#include <lodepng.h>
#include <string>
#include "common.hpp"
#include "cuda_common.hpp"
#include "h5read.h"
#include "standalone.h"
#include <vector>
#include <cstring>
#include <Eigen/Dense>
#include "xyz_to_rlp.cc"
#include "flood_fill.cc"
#include "sites_to_vecs.cc"
#include "fft3d.cc"
#include <chrono>
#include "simple_models.cc"

using Eigen::Vector3d;
using Eigen::Matrix3d;

int main(int argc, char **argv) {

    
    auto parser = CUDAArgumentParser();
    parser.add_h5read_arguments(); //will use h5 file to get metadata
    parser.add_argument("--images")
      .help("Maximum number of images to process")
      .metavar("NUM")
      .scan<'u', uint32_t>();
    auto args = parser.parse_args(argc, argv);


    std::unique_ptr<H5Read> reader_ptr;

    // Wait for read-readiness

    reader_ptr = args.file.empty() ? std::make_unique<H5Read>()
                                    : std::make_unique<H5Read>(args.file);
    // Bind this as a reference
    H5Read &reader = *reader_ptr;

    auto reader_mutex = std::mutex{};

    uint32_t num_images = parser.is_used("images") ? parser.get<uint32_t>("images")
                                                   : reader.get_number_of_images();


    // first get the Beam properties; wavelength and s0 vector
    auto wavelength_opt = reader.get_wavelength();
    if (!wavelength_opt) {
        printf(
            "Error: No wavelength provided. Please pass wavelength using: "
            "--wavelength\n");
        std::exit(1);
    }
    float wavelength_f = wavelength_opt.value();
    printf("INDEXER: Got wavelength from file: %f Å\n", wavelength_f);
    auto distance_opt = reader.get_detector_distance();
    if (!distance_opt) {
        printf("Error: No detector distance found in file.");
        std::exit(1);
    }
    float distance_f = distance_opt.value()*1000;
    printf("INDEXER: Got detector distance from file: %f mm\n", distance_f);
    auto pixel_size_f = reader.get_pixel_size();
    float pixel_size_x = pixel_size_f.value()[0]*1000;
    printf("INDEXER: Got detector pixel_size from file: %f mm\n", pixel_size_x);

    std::array<double, 3> module_offsets = reader.get_module_offsets();
    double origin_x = module_offsets[0] * 1000;
    double origin_y = module_offsets[1] * 1000;
    printf("INDEXER: Got detector origin x from file: %f mm\n", origin_x);
    printf("INDEXER: Got detector origin y from file: %f mm\n", origin_y);
    
    //TODO
    // Get metadata from file
    // implement max cell/d_min estimation. - will need annlib if want same result as dials.

    //FIXME don't assume s0 vector get from file
    //double wavelength = 0.9762535307519975;

    // now get detector properties
    // need fast, slow,  norm and origin, plus pixel size
    

    std::array<double, 3> fast_axis {1.0, 0.0, 0.0}; //FIXME get through reader - but const for I03 for now
    std::array<double, 3> slow_axis {0.0, -1.0, 0.0}; //FIXME get through reader
    // ^ change basis from nexus to iucr/imgcif convention (invert x and z)

    //std::array<double, 3> normal {0.0, 0.0, 0.0}; // fast_axis cross slow_axis
    //std::array<float, 3> origin {-75.61, 79.95, -150.0}; // FIXME
    //Vector3d origin {-75.61, 79.95, -150.0}; // FIXME
    //Vector3d origin {-153.60993960268158, 162.44624026693077, -200.45297988785603};
    Vector3d origin {-1.0*origin_x, origin_y, -1.0*distance_f};

    Matrix3d d_matrix{{fast_axis[0], slow_axis[0], origin[0]},{
        fast_axis[1], slow_axis[1], origin[1]},
        {fast_axis[2], slow_axis[2], origin[2]}};
    
    //FIXME remove assumption of pixel sizes being same in analysis code.
    //double pixel_size_x = pixel_size[0];//0.075;

    // Thickness not currently written to nxs file? Then need to calc mu from thickness.
    // Required to get equivalent results to dials.
    double mu = 3.9220780876;
    double t0 = 0.45;

    // now get scan properties e.g.
    int image_range_start = 1;
    double osc_start = 0.0;
    double osc_width = 0.10002778549596769;

    // finally gonio properties e.g.
    Matrix3d fixed_rotation{{1,0,0},{0,1,0},{0,0,1}};//{{0.965028,0.0598562,-0.255222},{-0.128604,-0.74028,-0.659883},{-0.228434,0.669628,-0.706694}};
    Vector3d rotation_axis {1.0,0.0,0.0};
    Matrix3d setting_rotation {{1,0,0},{0,1,0},{0,0,1}};
    auto t1 = std::chrono::system_clock::now();
    // Make the dxtbx-like models
    SimpleDetector detector(d_matrix, pixel_size_x, mu, t0, true);
    SimpleScan scan(image_range_start, osc_start, osc_width);
    SimpleGonio gonio(fixed_rotation, rotation_axis, setting_rotation);
    SimpleBeam beam(wavelength_f);

    // get processed reflection data from spotfinding
    std::string filename = "/dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24_rot/strong.refl";
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    std::vector<double> data = read_xyzobs_data(filename, array_name);

    std::vector<Vector3d> rlp = xyz_to_rlp(data, detector, beam, scan, gonio);

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

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    std::cout << "Total time for indexer: " << elapsed_time.count() << "s" << std::endl;

}