/**
 * @file pipe_handler.hpp
 * @brief Thread-safe pipe communication handler for outputting data.
 *
 * Provides a class for handling pipe-based communication to send
 * JSON data through a file descriptor in a thread-safe manner.
 */
#ifndef PIPE_HANDLER_HPP
#define PIPE_HANDLER_HPP

#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

using json = nlohmann::json;

/**
 * @brief Class for handling a pipe and sending data through it in a thread-safe manner.
 */
class PipeHandler {
  private:
    int pipe_fd;     ///< File descriptor for the pipe
    std::mutex mtx;  ///< Mutex for synchronization

  public:
    /**
     * @brief Constructor to initialize the PipeHandler object.
     * @param pipe_fd The file descriptor for the pipe.
     */
    PipeHandler(int pipe_fd) : pipe_fd(pipe_fd) {
        fmt::print("PipeHandler initialized with pipe_fd: {}\n", pipe_fd);
    }

    /**
     * @brief Destructor to close the pipe.
     */
    ~PipeHandler() {
        close(pipe_fd);
    }

    /**
     * @brief Sends data through the pipe in a thread-safe manner.
     * @param json_data A json object containing the data to be sent.
     */
    void sendData(const json &json_data) {
        // Lock the mutex, to ensure that only one thread writes to the pipe at a time
        // This unlocks the mutex when the function returns
        std::lock_guard<std::mutex> lock(mtx);

        // Convert the JSON object to a string
        std::string stringified_json = json_data.dump() + "\n";

        // Write the data to the pipe
        // Returns the number of bytes written to the pipe
        // Returns -1 if an error occurs
        ssize_t bytes_written =
          write(pipe_fd, stringified_json.c_str(), stringified_json.length());

        // Check if an error occurred while writing to the pipe
        if (bytes_written == -1) {
            std::cerr << "Error writing to pipe: " << strerror(errno) << std::endl;
        }
    }
};

#endif  // PIPE_HANDLER_HPP
