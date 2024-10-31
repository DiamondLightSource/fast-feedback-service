#include <iostream>
#include <lodepng.h>
#include <string>
#include "common.hpp"
#include "cuda_common.hpp"
#include "h5read.h"
#include "standalone.h"
#include <vector>
#include <cstring>

int main(int argc, char **argv) {
    auto parser = CUDAArgumentParser();
    parser.add_h5read_arguments(); //will use h5 file to get metadata
    parser.add_argument("--images")
      .help("Maximum number of images to process")
      .metavar("NUM")
      .scan<'u', uint32_t>();
    auto args = parser.parse_args(argc, argv);


    std::unique_ptr<Reader> reader_ptr;

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
    float wavelength = wavelength_opt.value();
    printf("INDEXER: Got wavelength from file: %f Å\n", wavelength);
    //FIXME don't assume s0 vector get from file
    std::array<float, 3> s0 {0.0, 0.0, -1.0/wavelength};

    // now get detector properties
    // need fast, slow,  norm and origin, plus pixel size
    //std::optional<std::array<float, 2>> pixel_size = reader.get_pixel_size();
    float pixel_size_x = 0.075;
     //FIXME remove assumption of pixel sizes being same in analysis code.

    std::array<float, 3> fast_axis {1.0, 0.0, 0.0}; //FIXME get through reader
    std::array<float, 3> slow_axis {0.0, 1.0, 0.0}; //FIXME get through reader
    // ^ change basis from nexus to iucr/imgcif convention (invert x and z)

    std::array<float, 3> normal {0.0, 0.0, 1.0}; // fast_axis cross slow_axis
    std::array<float, 3> origin {-75.61, 79.95, -150.0}; // FIXME
    //Vector3d origin {-75.61, 79.95, -150.0}; // FIXME

    // now get scan properties

    // get processed reflection data from spotfinding
    std::string filename = "/dls/mx-scratch/jbe/test_cuda_spotfinder/cm37235-2_ins_14_24/h5_file/strong.refl";
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    std::vector<double> data = read_xyzobs_data(filename, array_name);
    std::cout << data[0] << std::endl;
    std::cout << data[data.size()-1] << std::endl;
}