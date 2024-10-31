#include <iostream>
#include <string>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstring>
#include <chrono>
#include <vector>

using namespace std;

std::vector<double> read_xyzobs_data(string filename, string array_name){
    auto start_time = std::chrono::high_resolution_clock::now();
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, array_name.c_str(), H5P_DEFAULT);
    hid_t datatype = H5Dget_type(dataset);
    size_t datatype_size = H5Tget_size(datatype);
    hid_t dataspace = H5Dget_space(dataset);
    size_t num_elements = H5Sget_simple_extent_npoints(dataspace);
    hid_t space = H5Dget_space(dataset);
    std::vector<double> data_out(num_elements);

    H5Dread(dataset, datatype, H5S_ALL, space, H5P_DEFAULT, &data_out[0]);
    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - start_time)
        .count();
    std::cout << "XYZOBS READ TIME " << total_time << "s" << std::endl;
    
    H5Dclose(dataset);
    H5Fclose(file);
    return data_out;
}

