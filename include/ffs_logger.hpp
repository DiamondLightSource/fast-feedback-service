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
 * written to the same console and optional log file sink while
 * utilizing asynchronous logging for multi-threaded logging.
 */
class FFSLogger {
  public:
    // Get the singleton instance of the logger (thread-safe and lazy initialization)
    /**
     * @brief Retrieves the singleton instance of the logger.
     *
     * Lazily initialises and returns a reference to the shared logger
     * instance. This ensures a single point of access throughout the
     * application without requiring manual setup.
     */
    static spdlog::logger& getInstance() {
        if (!logger_) {
            initialiseLogger();
        }
        return *logger_;
    }

    /**
     * @brief Sets the logging level dynamically at runtime.
     * 
     * Allows modifying the logging level globally to control verbosity.
     *
     * @param level The desired logging level (e.g., spdlog::level::info, spdlog::level::debug).
     */
    static void setLevel(spdlog::level::level_enum level) {
        getInstance().set_level(level);
    }

  private:
    static std::shared_ptr<spdlog::logger> logger_;

    // Private constructor to prevent instantiation
    FFSLogger() = default;

    /**
     * @brief Creates and configures the logger instance.
     *
     * Internal helper that sets up the logger instance, including
     * sinks, formatting, and logging level. Called once on first use of
     * getInstance().
     */
    static void initialiseLogger() {
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
            logger_ = std::make_shared<spdlog::async_logger>(
              "FFSLogger",
              sinks.begin(),
              sinks.end(),
              spdlog::thread_pool(),
              spdlog::async_overflow_policy::block  // Block if queue is full
            );

            // Get logging level from environment variable (if set)
            const char* logLevelEnv = std::getenv("LOG_LEVEL");
            if (logLevelEnv) {
                logger_->set_level(spdlog::level::from_str(logLevelEnv));
            } else {
                logger_->set_level(spdlog::level::info);  // Default log level
            }

            // Register the logger globally
            spdlog::register_logger(logger_);
        } catch (const spdlog::spdlog_ex& ex) {
            throw std::runtime_error(std::string("Logger initialization failed: ")
                                     + ex.what());
        }
    }
};

// Global static access to the logger
inline auto& logger = FFSLogger::getInstance();