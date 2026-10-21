#include <bitshuffle.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <csignal>
#include <dx2/beam.hpp>
#include <dx2/beam_ops.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <experimental/mdspan>
#include <gemmi/unitcell.hpp>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stop_token>
#include <string>
#include <thread>

#include "../spotfinder/cbfread.hpp"
#include "../spotfinder/shmread.hpp"
#include "common.hpp"
#include "cuda_arg_parser.hpp"
#include "cuda_common.hpp"
#include "ffs_logger.hpp"
#include "h5read.h"
//#include "integrator/background.cuh"
//#include "integrator/background.hpp"
//#include "integrator/coordinate_system.hpp"
#include "integrator/extent.hpp"
//#include "integrator/lp_correction.hpp"
//#include "integrator/sigma_estimation.hpp"
#include "ellipsoid.cuh"
#include "math/device_precision.cuh"
#include "math/math_utils.cuh"
#include "math/vector3d.cuh"
//#include "predictor/predict.hpp"
#include "thread_count.hpp"
#include "version.hpp"
#include "integrator/ssx/ssx_integrate.hpp"
#include "mosaicity_parameterisation.hpp"
#include "predictor/predict.hpp"

using Eigen::Vector3d;

// Define a 2D mdspan type alias for convenience
using mdspan_2d =
  std::experimental::mdspan<scalar_t, std::experimental::dextents<size_t, 2>>;

// Global stop token for picking up user cancellation
std::stop_source global_stop;


// Function for passing to std::signal to register the stop request
extern "C" void stop_processing(int sig) {
    if (global_stop.stop_requested()) {
        // We already requested before, but we want it faster. Abort.
        std::quick_exit(1);
    } else {
        fmt::print("Running interrupted by user request\n");
        global_stop.request_stop();
    }
}

#pragma region Argument Parsing
class SSXIntegratorArgumentParser : public CUDAArgumentParser {
  public:
    SSXIntegratorArgumentParser(std::string version) : CUDAArgumentParser(version) {
        add_h5read_arguments();      // Override to use refl + expt
        add_integrator_arguments();  // Add integrator-specific args
    }

    void add_h5read_arguments() override {
        add_argument("--reflection", "-r")
          .metavar("strong.refl")
          .help("Input reflection table")
          .action([&](const std::string &value) { _reflection_filepath = value; });

        add_argument("--experiment", "-e")
          .metavar("experiments.expt")
          .help("Input experiment list")
          .action([&](const std::string &value) { _experiment_filepath = value; });

        _activated_h5read = true;
    }

    auto const reflections() const -> std::string {
        return _reflection_filepath;
    }
    auto const experiment() const -> std::string {
        return _experiment_filepath;
    }

  private:
    std::string _reflection_filepath;
    std::string _experiment_filepath;

    void add_integrator_arguments() {
        add_argument("-n", "--threads")
          .help("Number of parallel reader threads (default: 0 = auto-select)")
          .default_value<uint32_t>(0)
          .metavar("NUM")
          .scan<'u', uint32_t>();
    }
};
#pragma endregion Argument Parsing

#pragma region Application Entry
int main(int argc, char **argv) {
    logger.info("Version: {}", FFS_VERSION);

    // Parse arguments
    auto parser = SSXIntegratorArgumentParser(FFS_VERSION);
    auto args = parser.parse_args(argc, argv);
    const auto reflection_file = parser.reflections();
    const auto experiment_file = parser.experiment();
    float wait_timeout = 1.0;

    // Get threading parameters (0 = auto-select)
    uint32_t num_cpu_threads = parser.get<uint32_t>("threads");
    if (num_cpu_threads == 0) {
        num_cpu_threads = auto_select_thread_count();
        logger.info("Auto-selected {} CPU threads", num_cpu_threads);
    }
    logger.info("Running with {} CPU threads", num_cpu_threads);

    // Guard against missing files
    if (!std::filesystem::exists(reflection_file)) {
        logger.error("Reflection file not found: {}", reflection_file);
        return 1;
    }
    if (!std::filesystem::exists(experiment_file)) {
        logger.error("Experiment file not found: {}", experiment_file);
        return 1;
    }

    // Load reflection data
    logger.info("Loading data from file: {}", reflection_file);
    ReflectionTable reflections(reflection_file);

#pragma region Data preparation
    // Parse experiment list from JSON
    std::ifstream f(experiment_file);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error &ex) {
        logger.error("Failed to parse experiment file '{}': byte {}, {}",
                     experiment_file,
                     ex.byte,
                     ex.what());
        return 1;
    }

    // parse experiment list
    size_t num_reflections;
    mdspan_type<double> s1_vectors;
    auto s1_vectors_opt = reflections.column<double>("s1");
    if (!s1_vectors_opt) {
        logger.error("Column 's1' not found in reflection data.");
        return 1;
    }
    s1_vectors = *s1_vectors_opt;
    num_reflections = s1_vectors.extent(0);

    auto &beam = beam_ops::require_monochromatic(expt.beam());
    Eigen::Vector3d s0 = beam.get_s0();
    const Panel &panel = expt.detector().panels()[0];  // Assuming single panel detector
    const Crystal &crystal = expt.crystal();
    Eigen::Matrix3d A = crystal.get_A_matrix();

    // get set of unique ids, select out, loop over in parallel to refine and predict?

    // want a map of experiment identifier to image number
    std::unordered_map<std::string, int> identifier_to_image_number;
    auto scans = elist_json_obj["scan"];
    for (auto expt: elist_json_obj["experiment"]){
        std::string identifier = expt["identifier"];
        int scan_id = expt["scan"];
        int image_no = scans[scan_id]["image_range"][0];
        identifier_to_image_number[identifier] = image_no;
    }


    std::vector<ReflectionTable> split_tables;
    std::vector<int> ids;
    auto id_col = reflections.column<int>("id");
    if (!id_col){
        logger.error("Column 'id' not found in reflection data.");
        return 1;
    }
    ids = *id_col;
    for (auto i:reflections.get_experiment_ids()){
        std::vector<size_t> sel;
        for (int j=0;j<reflections.size();j++){
            if (ids[j] == i){
                sel.push_back(j);
            }
        }
        split_tables.push_back(reflections.select(sel));
    }

    std::vector<std::jthread> threads;
    for (int thread_id = 0; thread_id < num_cpu_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            // get the data from the table
            ReflectionTable& table = split_tables

            // refine predict
            /*Simple6MosaicityParameterisation model = refine_mosaicity(xyzcal_px,
                                                              xyzobs_px,
                                                              covariances,
                                                              intensities,
                                                              miller_indices,
                                                              mobs,
                                                              s0,
                                                              panel,
                                                              A);
            std::vector<Prediction> predictions = predict_ssx(model, s0, panel, A);*/

            // then save output predictions into a joint area to iterate over in integration.
        });
    }

    // Wait for all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }

    std::unordered_map<int, size_t> reflections_in_image;
    for (auto t : split_tables){
        std::vector<std::string> identifiers = t.get_identifiers();
        if (identifiers.size() != 1){
            logger.error("Reflection table selection went wrong.");
            return 1;
        }
        // now look up in experiment list to get image offset
        int image_no = identifier_to_image_number[identifier];
        reflections_in_image[image_no] = t.size();
    }

    /*std::vector<double> xyzcal_px;
    std::vector<double> xyzobs_px;
    std::vector<double> covariances;
    std::vector<int> miller_indices;
    std::vector<double> intensities;
    std::vector<double> mobs;
    auto xyzcal_px_opt = reflections.column<double>("xyzcal.px");
    if (!xyzcal_px_opt) {
        logger.error("Column 'xyzcal.px' not found in reflection data.");
        return 1;
    }
    xyzcal_px = std::vector<double>(
        xyzcal_px_opt.value().data_handle(),
        xyzcal_px_opt.value().data_handle() + xyzcal_px_opt.value().size());
    auto xyzobs_px_opt = reflections.column<double>("xyzobs.px.value");
    if (!xyzobs_px_opt) {
        logger.error("Column 'xyzobs.px.value' not found in reflection data.");
        return 1;
    }
    xyzobs_px = std::vector<double>(
        xyzobs_px_opt.value().data_handle(),
        xyzobs_px_opt.value().data_handle() + xyzobs_px_opt.value().size());
    auto hkl_vectors_opt = reflections.column<int>("miller_index");
    if (!hkl_vectors_opt) {
        logger.error("Column 'miller_index' not found in reflection data.");
        return 1;
    }
    miller_indices = std::vector<int>(
        hkl_vectors_opt.value().data_handle(),
        hkl_vectors_opt.value().data_handle() + hkl_vectors_opt.value().size());
    auto covariances_opt = reflections.column<double>("spot_covariance_2d_packed");
    if (!covariances_opt) {
        logger.error("Column 'covariances' not found in reflection data.");
        return 1;
    }
    covariances = std::vector<double>(
        covariances_opt.value().data_handle(),
        covariances_opt.value().data_handle() + covariances_opt.value().size());
    auto intensities_opt = reflections.column<double>("intensity.sum.value");
    if (!intensities_opt) {
        logger.error("Column 'intensity.sum.value' not found in reflection data.");
        return 1;
    }
    intensities = std::vector<double>(
        intensities_opt.value().data_handle(),
        intensities_opt.value().data_handle() + intensities_opt.value().size());
    auto mobs_opt = reflections.column<double>("spot_mobs_2d");
    if (!mobs_opt) {
        logger.error("Column 'spot_mobs_2d' not found in reflection data.");
        return 1;
    }
    mobs = std::vector<double>(
        mobs_opt.value().data_handle(),
        mobs_opt.value().data_handle() + mobs_opt.value().size());

    std::vector<Vector3d> xyzcal_vec;
    std::vector<Vector3d> xyzobs_vec;
    std::vector<Vector3d> covars_vec;
    std::vector<Vector3d> mobs_vec;
    for (std::size_t i=0;i<xyzcal_px.size() / 3;i++){
        xyzcal_vec.push_back({xyzcal_px[i], xyzcal_px[i+1], xyzcal_px[i+2]});
    }
    */
    


    /*const std::vector<Vector3d> &xyzcal_px,
                   const std::vector<Vector3d> &xyzobs_px,
                   const std::vector<Vector3d> &covariances,
                   const std::vector<double> &intensities,
                   const std::vector<Eigen::Vector3i> &miller_indices,
                   const std::vector<Vector2d> &mobs,
                   const Vector3d &s0,
                   const Panel &panel,
                   const Matrix3d &A) {*/
    /*Simple6MosaicityParameterisation model = refine_mosaicity(xyzcal_px,
                                                              xyzobs_px,
                                                              covariances,
                                                              intensities,
                                                              miller_indices,
                                                              mobs,
                                                              s0,
                                                              panel,
                                                              A);
    std::vector<Prediction> predictions = predict_ssx(model, s0, panel, A);*?

    /*mdspan_type<int> bbox_array;
    auto bbox_opt = reflections.column<int>("bbox");
    if (!bbox_opt) {
        logger.error("Column 'bbox' not found in reflection data.");
        return 1;
    }
    bbox_array = *bbox_opt;*/

    std::vector<BoundingBoxExtents> bboxes;
    for (std::size_t i=0;i<predictions.size();i++){
        /*BoundingBoxExtents bbox;
        bbox.x_min = bbox_array(i,0);
        bbox.x_max = bbox_array(i,1);
        bbox.y_min = bbox_array(i,2);
        bbox.y_max = bbox_array(i,3);
        bbox.z_min = bbox_array(i,4);
        bbox.z_max = bbox_array(i,5);*/
        bboxes.push_back(predictions[i].bbox_extent);
    }

    logger.info("Mapping reflections by image number (z layer)");
    std::unordered_map<int, std::vector<size_t>> reflections_by_image;
    // Per-image maximum (w+1)*(h+1) corner grid, which sizes the Kabsch
    // kernel's dynamic shared-memory corner tile for that image's launch.
    std::unordered_map<int, uint32_t> corner_tile_by_image;
    // need to split input reflection table, and match to image number through experiment list

    for (size_t refl_id = 0; refl_id < num_reflections; ++refl_id) {
        const auto &bbox = bboxes[refl_id];

        // The bbox is half-open in x and y, so w×h is exactly the pixel
        // count and the corner grid is one larger on each axis.
        const uint32_t corner_grid =
          static_cast<uint32_t>(bbox.x_max - bbox.x_min + 1)
          * static_cast<uint32_t>(bbox.y_max - bbox.y_min + 1);

        // Add this reflection to all images it spans. The shoebox z
        // range is half-open [z_min, z_max), so z_max is excluded.
        for (int z = bbox.z_min; z < bbox.z_max; ++z) {
            reflections_by_image[z].push_back(refl_id);
            uint32_t &image_max = corner_tile_by_image[z];
            image_max = std::max(image_max, corner_grid);
        }
    }

    logger.info("{} reflections mapped across {} unique images",
                num_reflections, reflections_by_image.size());

    //
    //file is nexus file extracted from elist
    std::unique_ptr<Reader> reader_ptr;
    std::string file = "/dls/i24/data/2026/mx36015-64/axcp/anaerobic/idris/idris_2492.nxs"; //FIXME
    reader_ptr = std::make_unique<H5Read>(file);
    Reader &reader = *reader_ptr;
    auto reader_mutex = std::mutex{};
    size_t bytes_per_pixel = reader.get_element_size();
    // Ensure this matches what we expect
    if (bytes_per_pixel != sizeof(pixel_t)) {
        fmt::print(
          "Error: Data type mismatch; This executable only accepts {} bit != {}\n",
          sizeof(pixel_t) * 8,
          bytes_per_pixel * 8);
        std::exit(bytes_per_pixel * 8);
    }
    uint32_t num_images = 1000;//reader.get_number_of_images();
    uint32_t height = reader.image_shape()[0];
    uint32_t width = reader.image_shape()[1];
    auto trusted_px_max = reader.get_trusted_range()[1];

    // Work out how many blocks this is
    dim3 gpu_thread_block_size{32, 16};
    dim3 blocks_dims{
      static_cast<unsigned int>(ceilf((float)width / gpu_thread_block_size.x)),
      static_cast<unsigned int>(ceilf((float)height / gpu_thread_block_size.y))};
    const int num_threads_per_block = gpu_thread_block_size.x * gpu_thread_block_size.y;
    const int num_blocks = blocks_dims.x * blocks_dims.y * blocks_dims.z;
    fmt::print("Image:       {:4d} x {:4d} = {} px\n", width, height, width * height);
    fmt::print("GPU Threads: {:4d} x {:<4d} = {}\n",
               gpu_thread_block_size.x,
               gpu_thread_block_size.y,
               num_threads_per_block);
    fmt::print("Blocks:      {:4d} x {:<4d} x {:2d} = {}\n",
               blocks_dims.x,
               blocks_dims.y,
               blocks_dims.z,
               num_blocks);
    fmt::print("Running with {} CPU threads\n", num_cpu_threads);


    auto all_images_start_time = std::chrono::high_resolution_clock::now();

    auto next_image = std::atomic<int>(0);
    auto completed_images = std::atomic<int>(0);

    auto cpu_sync = std::barrier{num_cpu_threads};
    double time_waiting_for_images = 0.0;

#pragma region Prep GPU Data Buffers
    // What info do we have that we need:
    // bbox
    // assumed panel_id = 0 i.e. single panel
    // computing mask - need s1, s2 from refl table
    //                - need s0, covariance matrix, panel

    /*Eigen::Matrix<scalar_t, 3, 3, Eigen::RowMajor> d_matrix_flat =
      panel.get_d_matrix().cast<scalar_t>();
    std::vector<fastvec::Vector3D> s1_vectors_vec(num_reflections);
    std::vector<fastvec::Vector3D> s2_vectors_vec(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        s1_vectors_vec[i] = to_vector3d(s1_vectors, i);
        s2_vectors_vec[i] = to_vector3d(s2_vectors, i);
    }*/
    // Allocate and copy GPU buffers (shared across all threads)
    /*DeviceBuffer<scalar_t> d_d_matrix(9);
    DeviceBuffer<fastvec::Vector3D> d_s1_vectors(num_reflections);
    DeviceBuffer<scalar_t> d_s2_vectors(num_reflections);*/
    DeviceBuffer<BoundingBoxExtents> d_bboxes(num_reflections);
    DeviceBuffer<uint8_t> d_success(num_reflections);

    /*d_d_matrix.assign(d_matrix_flat.data());
    d_s1_vectors.assign(s1_vectors_vec.data());
    d_s2_vectors.assign(s2_vectors_vec.data());*/
    // Test case, let's just loop over bboxes summing all pixels.
    d_bboxes.assign(bboxes.data());
    std::vector<uint8_t> success_init(num_reflections, 1);
    d_success.assign(success_init.data());

    // Upload the detector mask once (constant across all images).
    // non-zero = valid, 0 = masked. If the reader provides no mask,
    // treat every pixel as valid. Stored flat (width*height), indexed
    // `gy * width + gx`.
    std::vector<uint8_t> h_mask(static_cast<size_t>(width) * height, 1);
    if (auto mask_opt = reader.get_mask()) {
        const auto &mask_span = *mask_opt;
        std::copy(mask_span.begin(), mask_span.end(), h_mask.begin());
        logger.info("Using detector mask from reader ({} pixels)", mask_span.size());
    } else {
        logger.info("No detector mask available; treating all pixels as valid");
    }
    DeviceBuffer<uint8_t> d_mask(static_cast<size_t>(width) * height);
    d_mask.assign(h_mask.data());

    //DetectorParameters det_params = make_detector_params(panel);
    
    //fastvec::Vector3D s0_vec = to_vector3d(s0);

    DeviceBuffer<accumulator_t> d_foreground_sum(num_reflections);
    DeviceBuffer<uint32_t> d_foreground_count(num_reflections);


    cudaMemset(d_foreground_sum.data(), 0, num_reflections * sizeof(accumulator_t));
    cudaMemset(d_foreground_count.data(), 0, num_reflections * sizeof(uint32_t));
    cuda_throw_error();

    // assume already calculated corrections (LP, qe)
#pragma endregion Prep GPU Data Buffers

#pragma region Thread launch
    // We are going to iterate through the images,
    // doing refine, predict and then integrate on gpu
    // The result we will store is the reflection table (and mosaicity models?).
    // For testing, assume we have already refined and predicted.

    // Spawn the reader threads
    std::vector<std::jthread> threads;
    for (int thread_id = 0; thread_id < num_cpu_threads; ++thread_id) {
        threads.emplace_back([&, thread_id]() {
            auto stop_token = global_stop.get_token();
            CudaStream stream;

            auto host_image = make_cuda_pinned_malloc<pixel_t>(width * height);
            //auto host_results = make_cuda_pinned_malloc<uint8_t>(width * height);
            auto device_image = PitchedMalloc<pixel_t>(width, height);
            /*auto device_results =
              PitchedMalloc<uint8_t>(make_cuda_malloc<uint8_t[]>(mask.pitch * height),
                                     width,
                                     height,
                                     mask.pitch);*/

            // Buffer for reading compressed chunk data in
            auto raw_chunk_buffer =
              std::vector<uint8_t>(width * height * sizeof(pixel_t));

            // Let all threads do setup tasks before reading starts
            cpu_sync.arrive_and_wait();
            //CudaEvent start, copy, post, postcopy, end;

            // Get the time the lastimage was received to avoid waiting for too long
            auto last_image_received = std::chrono::high_resolution_clock::now();

            while (!stop_token.stop_requested()) {
                auto image_num = next_image.fetch_add(1);
                if (image_num >= num_images) {
                    break;
                }
                // Check if this image has any reflections
                if (reflections_by_image.find(image_num)
                    == reflections_by_image.end()) {
                    completed_images += 1;
                    continue;
                }

                {
                    // Do we really need the wait check in integration?
                    std::scoped_lock lock(reader_mutex);
                    auto swmr_wait_start_time =
                      std::chrono::high_resolution_clock::now();

                    // Check that our image is available and wait if not
                    while (!reader.is_image_available(image_num)
                           && !stop_token.stop_requested()) {
                        auto current_time = std::chrono::high_resolution_clock::now();
                        auto elapsed_wait_time =
                          std::chrono::duration_cast<std::chrono::duration<double>>(
                            current_time - last_image_received)
                            .count();

                        if (elapsed_wait_time > wait_timeout) {
                            logger.warn("Timeout waiting for image {}",
                                        image_num);
                            global_stop.request_stop();
                            break;
                        }

                        // Sleep for a bit to avoid busy-waiting
                        //std::this_thread::sleep_for(100ms);
                    }

                    if (stop_token.stop_requested()) {
                        break;
                    }

                    // If the image is available, update the last image received time
                    last_image_received = std::chrono::high_resolution_clock::now();

                    time_waiting_for_images +=
                      std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now()
                        - swmr_wait_start_time)
                        .count();
                }
                // Sized buffer for the actual data read from file
                std::span<uint8_t> buffer;
                // Fetch the image data from the reader.. loop in case we need to wait
                while (true) {
                    {
                        std::scoped_lock lock(reader_mutex);
                        buffer =
                          reader.get_raw_chunk(image_num, raw_chunk_buffer);
                    }
                    // /dev/shm we might not have an atomic write
                    if (buffer.size() == 0) {
                        logger.warn(
                          "\033[1mRace Condition?!?? Got buffer size 0 for image "
                          "{}. Sleeping.\033[0m",
                          image_num);
                        //std::this_thread::sleep_for(100ms);
                        continue;
                    }
                    break;
                }

#pragma region Decompression
                // Decompress this data, outside of the mutex.
                // We do this here rather than in the reader, because we
                // anticipate that we will want to eventually offload
                // the decompression
                switch (reader.get_raw_chunk_compression()) {
                case Reader::ChunkCompression::BITSHUFFLE_LZ4:
                    bshuf_decompress_lz4(buffer.data() + 12,
                                         host_image.get(),
                                         width * height,
                                         sizeof(pixel_t),
                                         0);
                    break;
                case Reader::ChunkCompression::BYTE_OFFSET_32:
                    decompress_byte_offset<pixel_t>(
                      buffer,
                      {host_image.get(),
                       static_cast<std::span<pixel_t>::size_type>(width * height)});
                    break;
                }

                //start.record(stream);
                // Copy the image to GPU
                cudaMemcpy2DAsync(device_image.get(),
                                             device_image.pitch_bytes(),
                                             host_image.get(),
                                             width * sizeof(pixel_t),
                                             width * sizeof(pixel_t),
                                             height,
                                             cudaMemcpyHostToDevice,
                                             stream);
                cuda_throw_error();
                //copy.record(stream);
#pragma endregion Decompression
                // Get reflection indices for this image and copy to device
                const auto &refl_indices = reflections_by_image[image_num];
                size_t num_refls_this_image = refl_indices.size();

                // Allocate per-image device buffer for reflection indices
                DeviceBuffer<size_t> d_reflection_indices(num_refls_this_image);
                d_reflection_indices.assign(refl_indices.data());
                logger.info("Launching GPU kernel for image {} with {} reflections",
                            image_num,
                            num_refls_this_image);
                // For test, just sum all pixels in bbox.
                compute_ellipsoid_integrate(device_image.get(),
                                    device_image.pitch_bytes(),
                                    width,
                                    height,
                                    image_num,
                                    d_bboxes.data(),
                                    d_reflection_indices.data(),
                                    num_refls_this_image,
                                    d_mask.data(),
                                    corner_tile_by_image.at(image_num),
                                    d_foreground_sum.data(),
                                    d_foreground_count.data(),
                                    d_success.data(),
                                    stream);

                logger.trace("Thread {} loaded image {}", thread_id, image_num);
                completed_images += 1;
            }
        });
    }
    // Wait for all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }

    float total_time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - all_images_start_time)
        .count();

    logger.info("{} images processed in {:.2f} s ({:.1f} fps)",
                int(completed_images),
                total_time,
                completed_images / total_time);
    if (time_waiting_for_images < 10) {
        logger.info("Total time waiting for images: {:.0f} ms",
                    time_waiting_for_images * 1000);
    } else {
        logger.info("Total time waiting for images: {:.2f} s", time_waiting_for_images);
    }
    cudaDeviceSynchronize();
    cuda_throw_error();
    logger.info("Finalizing integration for {} reflections", num_reflections);
    // Allocate host buffers for results
    std::vector<accumulator_t> h_foreground_sum(num_reflections);
    std::vector<uint32_t> h_foreground_count(num_reflections);
    // Copy results from device to host
    d_foreground_sum.extract(h_foreground_sum.data());
    d_foreground_count.extract(h_foreground_count.data());

    // Compute final intensities for each reflection
    std::vector<double> intensities(num_reflections);
    for (size_t i = 0; i < num_reflections; ++i) {
        double fg_sum = h_foreground_sum[i];
        intensities[i] = fg_sum;
    }
    logger.info("Building output reflection table");
    // Build output reflection table.
    std::vector<std::string> identifiers = reflections.get_identifiers();
    std::vector<uint64_t> ids_in = reflections.get_experiment_ids();
    ReflectionTable integrated_data(ids_in, identifiers);
    integrated_data.add_column("intensity.sum.value", num_reflections, 1, intensities);
    integrated_data.write("integrated.refl");

    return 0;
}