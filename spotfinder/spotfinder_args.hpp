/**
 * @file spotfinder_args.hpp
 * @brief Argument parser for the spotfinder application.
 *
 * Extends the CUDA argument parser with spotfinder-specific
 * command-line arguments for image processing configuration.
 */
#ifndef SPOTFINDER_ARGS_HPP
#define SPOTFINDER_ARGS_HPP

#include "cuda_arg_parser.hpp"

/**
 * @brief Argument parser for the spotfinder application.
 *
 * Extends CUDAArgumentParser with spotfinder-specific arguments
 * including image processing options, algorithm selection, and
 * resolution filtering parameters.
 */
class SpotfinderArgumentParser : public CUDAArgumentParser {
  public:
    SpotfinderArgumentParser(std::string version) : CUDAArgumentParser(version) {
        add_h5read_arguments();
        add_spotfinder_arguments();
    }

    /// Override the HDF5 reading arguments
    void add_h5read_arguments() override {
        // Check if implicit sample is enable via environment variable
        bool implicit_sample = std::getenv("H5READ_IMPLICIT_SAMPLE") != nullptr;
        // Create a mutually exclusive group for sample vs file input
        auto &group = add_mutually_exclusive_group(!implicit_sample);

        group.add_argument("--sample")
          .help("Use generated test data (H5READ_IMPLICIT_SAMPLE)")
          .implicit_value(true);

        group.add_argument("file")
          .metavar("FILE.nxs")
          .help("Path to Nexus file")
          .action([&](const std::string &val) { _filepath = val; });

        _activated_h5read = true;
    }

    auto const file() const -> const std::string & {
        if (!_activated_h5read) {
            throw std::runtime_error("HDF5 reading arguments not activated");
        }
        return _filepath;
    }

    void add_spotfinder_arguments() {
        add_argument("-n", "--threads")
          .help("Number of parallel reader threads")
          .default_value<uint32_t>(1)
          .metavar("NUM")
          .scan<'u', uint32_t>();

        add_argument("--validate")
          .help("Run DIALS standalone validation")
          .default_value(false)
          .implicit_value(true);

        add_argument("--images")
          .help("Maximum number of images to process")
          .metavar("NUM")
          .scan<'u', uint32_t>();

        add_argument("--writeout")
          .help("Write diagnostic output images")
          .default_value(false)
          .implicit_value(true);

        add_argument("--min-spot-size")
          .help("2D Reflections with a pixel count below this will be discarded.")
          .metavar("N")
          .default_value<uint32_t>(3)
          .scan<'u', uint32_t>();

        add_argument("--min-spot-size-3d")
          .help("3D Reflections with a pixel count below this will be discarded.")
          .metavar("N")
          .default_value<uint32_t>(3)
          .scan<'u', uint32_t>();

        add_argument("--max-peak-centroid-separation")
          .help(
            "Reflections with a peak-centroid difference greater than this will be "
            "filtered during output.")
          .metavar("N")
          .default_value<float>(2.0)
          .scan<'f', float>();

        add_argument("--start-index")
          .help(
            "Index of first image. Only used for CBF reading, and can only be 0 or 1.")
          .metavar("N")
          .default_value<uint32_t>(0)
          .scan<'u', uint32_t>();

        add_argument("-t", "--timeout")
          .help("Amount of time (in seconds) to wait for new images before failing.")
          .metavar("S")
          .default_value<float>(30)
          .scan<'f', float>();

        add_argument("-fd", "--pipe_fd")
          .help("File descriptor for the pipe to output data through")
          .metavar("FD")
          .default_value<int>(-1)
          .scan<'i', int>();

        add_argument("-a", "--algorithm")
          .help("Dispersion algorithm to use")
          .metavar("ALGO")
          .default_value<std::string>("dispersion");

        add_argument("--dmin")
          .help("Minimum resolution (Å)")
          .metavar("MIN D")
          .default_value<float>(-1.f)
          .scan<'f', float>();

        add_argument("--dmax")
          .help("Maximum resolution (Å)")
          .metavar("MAX D")
          .default_value<float>(-1.f)
          .scan<'f', float>();

        add_argument("-w", "-λ", "--wavelength")
          .help("Wavelength of the X-ray beam (Å)")
          .metavar("λ")
          .scan<'f', float>();

        add_argument("--detector").help("Detector geometry JSON").metavar("JSON");

        add_argument("-h5", "--save-h5")
          .help("Save the output to an HDF5 file")
          .metavar("FILE")
          .default_value(false)
          .implicit_value(true);

        add_argument("--output-for-index")
          .help("Pipe spot centroids from 2D images to enable indexing")
          .default_value(false)
          .implicit_value(true);
    }

  private:
    std::string _filepath;  ///< Path to the input file
};

#endif  // SPOTFINDER_ARGS_HPP
