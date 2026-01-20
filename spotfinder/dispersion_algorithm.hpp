/**
 * @file dispersion_algorithm.hpp
 * @brief Dispersion algorithm enumeration and string conversion.
 *
 * Defines the supported dispersion algorithms for spotfinding
 * and provides case-insensitive parsing from string input.
 */
#ifndef DISPERSION_ALGORITHM_HPP
#define DISPERSION_ALGORITHM_HPP

#include <algorithm>
#include <stdexcept>
#include <string>

/**
 * @brief Struct to store the dispersion algorithm and its string representation.
 */
struct DispersionAlgorithm {
    std::string algorithm_str;
    enum class Algorithm { DISPERSION, DISPERSION_EXTENDED };
    Algorithm algorithm;

    /**
     * @brief Constructor to initialize the DispersionAlgorithm object.
     * @param input The string representation of the algorithm.
     * @throws std::invalid_argument if the algorithm string is not recognized.
     */
    DispersionAlgorithm(std::string input) {
        // Convert the input to lowercase for case-insensitive comparison
        this->algorithm_str = input;
        std::transform(input.begin(), input.end(), input.begin(), ::tolower);
        if (input == "dispersion") {
            this->algorithm_str = "Dispersion";
            this->algorithm = Algorithm::DISPERSION;
        } else if (input == "dispersion_extended") {
            this->algorithm_str = "Dispersion Extended";  // ✨
            this->algorithm = Algorithm::DISPERSION_EXTENDED;
        } else {
            throw std::invalid_argument("Invalid algorithm specified");
        }
    }
};

#endif  // DISPERSION_ALGORITHM_HPP
