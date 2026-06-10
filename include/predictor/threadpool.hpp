/**
 * @file threadpool.hpp
 * @brief Minimal fixed-size thread pool used by the spot predictor.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
  public:
    ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Enqueue a task
    void enqueue(std::function<void()> task);

  private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};
