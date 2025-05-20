#pragma once

#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <memory>

/**
 * @brief Thread-safe singleton logger class.
 * 
 * This class provides a single point of access to the logger instance
 * throughout the application, ensuring that all log messages are
 * written to the same log file and console sink while utilizing
 * asynchronous logging for multi-threaded logging.
 */
class FFSLogger {
  public:
    // Get the singleton instance of the logger (thread-safe and lazy initialization)
    /**
     * @brief Retrieves the singleton instance of the logger.
     * 
     * This method uses lazy initialization to initialize the logger on
     * the first call and returns the same instance on subsequent calls,
     * ensuring a single point of access throughout the application.
     *
     * @return std::shared_ptr<spdlog::logger>& A shared pointer to the logger instance.
     */
    static std::shared_ptr<spdlog::logger>& getInstance() {
        static std::shared_ptr<spdlog::logger> instance = createLogger();
        return instance;
    }

    /**
     * @brief Sets the logging level dynamically at runtime.
     * 
     * Allows modifying the logging level globally to control verbosity.
     *
     * @param level The desired logging level (e.g., spdlog::level::info, spdlog::level::debug).
     */
    static void setLevel(spdlog::level::level_enum level) {
        getInstance()->set_level(level);
    }

  private:
    // Private constructor to prevent instantiation
    FFSLogger() = default;

    /**
     * @brief Creates and configures the logger instance.
     * 
     * Initializes the logger with a rotating file sink and a colored console sink,
     * enabling asynchronous logging.
     *
     * @return std::shared_ptr<spdlog::logger> The configured logger instance.
     */
    static std::shared_ptr<spdlog::logger> createLogger() {
        try {
            // Queue size for async messages
            size_t queue_size = 8192;  // Can adjust based on app requirements

            // Initialize spdlog asynchronous mode with a background worker thread
            spdlog::init_thread_pool(queue_size, 1);

            // Create a rotating file sink (max 5MB per file, 3 rotated files)
            // auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            //   "ffs_log.txt", 5 * 1024 * 1024, 3);
            // file_sink->set_pattern(
            //   "[%Y-%m-%d %H:%M:%S] [PID:%P Thread:%t] [%^%l%$] [%s:%#] %v");

            // Create a colored console sink
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [thread %t] [%^%l%$] %v");

            // Combine sinks into one asynchronous logger
            // std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
            std::vector<spdlog::sink_ptr> sinks{console_sink};
            auto async_logger = std::make_shared<spdlog::async_logger>(
              "FFSLogger",
              sinks.begin(),
              sinks.end(),
              spdlog::thread_pool(),
              spdlog::async_overflow_policy::block  // Block if queue is full
            );

            // Get logging level from environment variable (if set)
            const char* logLevelEnv = std::getenv("LOG_LEVEL");
            if (logLevelEnv) {
                async_logger->set_level(spdlog::level::from_str(logLevelEnv));
            } else {
                async_logger->set_level(spdlog::level::info);  // Default log level
            }

            // Register the logger globally
            spdlog::register_logger(async_logger);

            return async_logger;
        } catch (const spdlog::spdlog_ex& ex) {
            throw std::runtime_error(std::string("Logger initialization failed: ")
                                     + ex.what());
        }
    }
};