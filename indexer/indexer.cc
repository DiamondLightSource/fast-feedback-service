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
#include "fft3d.cc"
#include <chrono>

using Eigen::Vector3d;
using Eigen::Matrix3d;

int main(int argc, char **argv) {

    auto t1 = std::chrono::system_clock::now();
    auto parser = CUDAArgumentParser();
    /*parser.add_h5read_arguments(); //will use h5 file to get metadata
    parser.add_argument("--images")
      .help("Maximum number of images to process")
      .metavar("NUM")
      .scan<'u', uint32_t>();*/
    auto args = parser.parse_args(argc, argv);


    /*std::unique_ptr<Reader> reader_ptr;

    // Wait for read-readiness

    reader_ptr = args.file.empty() ? std::make_unique<H5Read>()
                                    : std::make_unique<H5Read>(args.file);
    // Bind this as a reference
    Reader &reader = *reader_ptr;

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
    printf("INDEXER: Got wavelength from file: %f Å\n", wavelength_f);*/
    //FIXME don't assume s0 vector get from file
    double wavelength = 0.976254;
    Vector3d s0 {0.0, 0.0, -1.0/wavelength};

    // now get detector properties
    // need fast, slow,  norm and origin, plus pixel size
    //std::optional<std::array<float, 2>> pixel_size = reader.get_pixel_size();
    float pixel_size_x = 0.075;
     //FIXME remove assumption of pixel sizes being same in analysis code.

    std::array<float, 3> fast_axis {1.0, 0.0, 0.0}; //FIXME get through reader
    std::array<float, 3> slow_axis {0.0, -1.0, 0.0}; //FIXME get through reader
    // ^ change basis from nexus to iucr/imgcif convention (invert x and z)

    std::array<float, 3> normal {0.0, 0.0, 1.0}; // fast_axis cross slow_axis
    //std::array<float, 3> origin {-75.61, 79.95, -150.0}; // FIXME
    //Vector3d origin {-75.61, 79.95, -150.0}; // FIXME
    Vector3d origin {-153.61, 162.446, -200.453};

    Matrix3d d_matrix{{fast_axis[0], slow_axis[0], normal[0]+origin[0],
        fast_axis[1], slow_axis[1], normal[1]+origin[1],
        fast_axis[2], slow_axis[2], normal[2]+origin[2]}};
    // now get scan properties e.g.
    int image_range_start = 0;
    double osc_start = 0.0;
    double osc_width = 0.1;

    // finally gonio properties e.g.
    Matrix3d fixed_rotation{{1,0,0},{0,1,0},{0,0,1}};//{{0.965028,0.0598562,-0.255222},{-0.128604,-0.74028,-0.659883},{-0.228434,0.669628,-0.706694}};
    Vector3d rotation_axis {1.0,0.0,0.0};
    Matrix3d setting_rotation {{1,0,0},{0,1,0},{0,0,1}};

    // get processed reflection data from spotfinding
    std::string filename = "/dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24_rot/strong.refl";
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    std::vector<double> data = read_xyzobs_data(filename, array_name);

    std::vector<Vector3d> rlp = xyz_to_rlp(
        data, fixed_rotation, d_matrix, wavelength, pixel_size_x,image_range_start,
        osc_start, osc_width, rotation_axis, setting_rotation);

    std::cout << data[0] << std::endl;
    std::cout << data[data.size()-1] << std::endl;
    std::cout << rlp[0][0] << std::endl;
    std::cout << rlp[rlp.size()-1][0] << std::endl;
    std::cout << "Number of reflections: " << rlp.size() << std::endl;

    std::vector<double> real_fft(256*256*256);
    std::vector<bool> used_in_indexing = fft3d(rlp, real_fft, 1.8);
    auto t2 = std::chrono::system_clock::now();
    std::cout << real_fft[0] << " " << used_in_indexing[0] << std::endl;
    std::chrono::duration<double> elapsed_time = t2 - t1;
    std::cout << "Total time for indexer: " << elapsed_time.count() << "s" << std::endl;

}