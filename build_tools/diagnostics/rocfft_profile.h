// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

// Opt-in host-only instrumentation. RTC subprocesses never start a Session.
// Timers overlap across threads and nesting; do not sum them as elapsed time.
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#endif

namespace rocfft_build_profile {
using Clock = std::chrono::steady_clock;
enum Phase {
  queue_wait,
  get_mutex_wait,
  get_mutex_hold,
  get_query,
  get_copy,
  generate,
  semaphore_wait,
  subprocess_compile,
  inprocess_lock_wait,
  inprocess_compile,
  store_mutex_wait,
  store_mutex_hold,
  store_query,
  pending_mutex_wait,
  pending_mutex_hold,
  future_wait,
  producer_stockham,
  producer_realcomplex,
  producer_twiddle,
  producer_solution,
  workers_join,
  final_copy,
  cleanup,
  teardown,
  db_initialize,
  aot_begin,
  aot_commit,
  COUNT
};
inline constexpr const char *names[] = {"queue_wait",
                                        "get_mutex_wait",
                                        "get_mutex_hold",
                                        "get_query",
                                        "get_copy",
                                        "generate",
                                        "semaphore_wait",
                                        "subprocess_compile",
                                        "inprocess_lock_wait",
                                        "inprocess_compile",
                                        "store_mutex_wait",
                                        "store_mutex_hold",
                                        "store_query",
                                        "pending_mutex_wait",
                                        "pending_mutex_hold",
                                        "future_wait",
                                        "producer_stockham",
                                        "producer_realcomplex",
                                        "producer_twiddle",
                                        "producer_solution",
                                        "workers_join",
                                        "final_copy",
                                        "cleanup",
                                        "teardown",
                                        "db_initialize",
                                        "aot_begin",
                                        "aot_commit"};
static_assert(sizeof(names) / sizeof(names[0]) == COUNT);
struct Metric {
  std::atomic<unsigned long long> count{0}, ns{0}, max_ns{0}, active{0};
};
struct State {
  std::atomic<bool> enabled{false};
  std::array<Metric, COUNT> metrics;
  std::atomic<unsigned long long> queued{0}, popped{0}, queue_depth{0};
  std::atomic<unsigned long long> cache_hits{0}, cache_misses{0},
      source_bytes{0}, code_bytes{0};
  std::atomic<unsigned long long> subprocess_failures{0}, fallbacks{0},
      pending_shared{0};
  std::atomic<unsigned long long> pending_new{0}, semaphore_capacity{0},
      store_errors{0};
  std::atomic<unsigned long long> get_rows{0}, get_other{0}, get_busy{0},
      store_busy{0};
  std::atomic<unsigned long long> pipe_errors{0}, last_pipe_errno{0};
  std::atomic<unsigned long long> spawn_errors{0}, last_spawn_errno{0},
      wait_errors{0}, last_wait_errno{0};
  std::atomic<unsigned long long> nonzero_exits{0}, last_child_exit{0},
      child_signals{0}, last_child_signal{0};
  std::mutex detail_mutex;
  std::vector<std::pair<std::string, std::string>> properties;
  std::vector<std::string> failures;
};
inline State &state() {
  static State value;
  return value;
}
inline bool enabled() {
  return state().enabled.load(std::memory_order_relaxed);
}
inline void add(std::atomic<unsigned long long> &value,
                unsigned long long n = 1) {
  if (enabled())
    value.fetch_add(n, std::memory_order_relaxed);
}
inline std::string json_quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32 || c > 126)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<int>(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
inline void property(const std::string &key, const std::string &value) {
  if (!enabled())
    return;
  std::lock_guard<std::mutex> lock(state().detail_mutex);
  state().properties.emplace_back(key, value);
}
inline void error_code(std::atomic<unsigned long long> &count,
                       std::atomic<unsigned long long> &last, int code) {
  if (!enabled())
    return;
  count.fetch_add(1, std::memory_order_relaxed);
  last.store(code, std::memory_order_relaxed);
}
inline void failure(const char *value) {
  if (!enabled())
    return;
  add(state().subprocess_failures);
  // Bound both collection and text. Never emit generated source or entire
  // compiler logs.
  std::string text;
  for (size_t i = 0; value && value[i] && i < 160; ++i)
    text += value[i];
  std::lock_guard<std::mutex> lock(state().detail_mutex);
  if (state().failures.size() < 8)
    state().failures.push_back(text);
}
class Scope {
  Metric *metric = nullptr;
  Clock::time_point start;

public:
  explicit Scope(Phase phase) {
    if (enabled()) {
      metric = &state().metrics[phase];
      start = Clock::now();
      metric->active.fetch_add(1, std::memory_order_relaxed);
    }
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
  ~Scope() {
    if (!metric)
      return;
    auto ns = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                             start)
            .count());
    metric->ns.fetch_add(ns, std::memory_order_relaxed);
    metric->count.fetch_add(1, std::memory_order_relaxed);
    auto previous = metric->max_ns.load(std::memory_order_relaxed);
    while (previous < ns && !metric->max_ns.compare_exchange_weak(
                                previous, ns, std::memory_order_relaxed)) {
    }
    metric->active.fetch_sub(1, std::memory_order_relaxed);
  }
};
// The same mutex is acquired/released in the same scope; only timing is added.
class Lock {
  std::unique_lock<std::mutex> lock;
  Scope held;
  static std::unique_lock<std::mutex> acquire(std::mutex &mutex, Phase wait) {
    Scope scope(wait);
    return std::unique_lock<std::mutex>(mutex);
  }

public:
  Lock(std::mutex &mutex, Phase wait, Phase hold)
      : lock(acquire(mutex, wait)), held(hold) {}
};
inline long long thread_cpu_ns() {
#ifndef _WIN32
  timespec t{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0)
    return static_cast<long long>(t.tv_sec) * 1000000000LL + t.tv_nsec;
#endif
  return -1;
}
struct CpuUsage {
  bool available = false;
  double user_s = 0, system_s = 0;
  long maxrss_kib = 0;
  int error = -1;
};
inline CpuUsage cpu_usage(bool children) {
  CpuUsage result;
#ifndef _WIN32
  rusage value{};
  if (getrusage(children ? RUSAGE_CHILDREN : RUSAGE_SELF, &value) == 0) {
    result.available = true;
    result.error = 0;
    result.user_s = value.ru_utime.tv_sec + value.ru_utime.tv_usec / 1000000.0;
    result.system_s =
        value.ru_stime.tv_sec + value.ru_stime.tv_usec / 1000000.0;
    result.maxrss_kib = value.ru_maxrss;
  } else
    result.error = errno;
#endif
  return result;
}
inline void write_usage(std::ostream &out, const CpuUsage &start,
                        const CpuUsage &current) {
  out << "{\"available\":"
      << (start.available && current.available ? "true" : "false")
      << ",\"start_errno\":" << start.error
      << ",\"current_errno\":" << current.error;
  if (start.available && current.available)
    out << ",\"start_user_s\":" << start.user_s
        << ",\"start_system_s\":" << start.system_s
        << ",\"current_user_s\":" << current.user_s
        << ",\"current_system_s\":" << current.system_s
        << ",\"user_delta_s\":" << current.user_s - start.user_s
        << ",\"system_delta_s\":" << current.system_s - start.system_s
        << ",\"start_maxrss_kib\":" << start.maxrss_kib
        << ",\"current_maxrss_kib\":" << current.maxrss_kib;
  out << '}';
}
class Session {
  std::ofstream file;
  std::string arch, database, output;
  Clock::time_point start;
  std::thread sampler;
  std::mutex stop_mutex;
  std::condition_variable stop_cv;
  bool stop = false;
  long long sampler_cpu = 0;
  unsigned long long snapshots = 0;
  CpuUsage initial_self, initial_children;
  void snapshot(const char *event) {
    auto &s = state();
    auto wall = std::chrono::duration<double>(Clock::now() - start).count();
    auto epoch = std::chrono::duration<double>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
    file << "{\"event\":" << json_quote(event)
         << ",\"arch\":" << json_quote(arch) << ",\"epoch_s\":" << std::fixed
         << std::setprecision(6) << epoch << ",\"relative_s\":" << wall
         << ",\"steady_since_epoch_s\":"
         << std::chrono::duration<double>(Clock::now().time_since_epoch())
                .count()
         << ",\"snapshot_sequence\":" << snapshots++
         << ",\"sampler_cpu_ns\":" << sampler_cpu << ",\"phases\":{";
    for (size_t i = 0; i < COUNT; ++i) {
      if (i)
        file << ',';
      auto &m = s.metrics[i];
      file << json_quote(names[i]) << ":{\"active\":" << m.active.load()
           << ",\"completed\":" << m.count.load()
           << ",\"sum_ns\":" << m.ns.load() << ",\"max_ns\":" << m.max_ns.load()
           << '}';
    }
    file << "},\"counters\":{";
#define ROCFFT_PROFILE_COUNTER(name)                                           \
  file << "\"" #name "\":" << s.name.load() << ','
    ROCFFT_PROFILE_COUNTER(queued);
    ROCFFT_PROFILE_COUNTER(popped);
    ROCFFT_PROFILE_COUNTER(queue_depth);
    ROCFFT_PROFILE_COUNTER(cache_hits);
    ROCFFT_PROFILE_COUNTER(cache_misses);
    ROCFFT_PROFILE_COUNTER(source_bytes);
    ROCFFT_PROFILE_COUNTER(code_bytes);
    ROCFFT_PROFILE_COUNTER(subprocess_failures);
    ROCFFT_PROFILE_COUNTER(fallbacks);
    ROCFFT_PROFILE_COUNTER(pending_shared);
    ROCFFT_PROFILE_COUNTER(pending_new);
    ROCFFT_PROFILE_COUNTER(semaphore_capacity);
    ROCFFT_PROFILE_COUNTER(get_rows);
    ROCFFT_PROFILE_COUNTER(get_other);
    ROCFFT_PROFILE_COUNTER(get_busy);
    ROCFFT_PROFILE_COUNTER(store_busy);
    ROCFFT_PROFILE_COUNTER(pipe_errors);
    ROCFFT_PROFILE_COUNTER(last_pipe_errno);
    ROCFFT_PROFILE_COUNTER(spawn_errors);
    ROCFFT_PROFILE_COUNTER(last_spawn_errno);
    ROCFFT_PROFILE_COUNTER(wait_errors);
    ROCFFT_PROFILE_COUNTER(last_wait_errno);
    ROCFFT_PROFILE_COUNTER(nonzero_exits);
    ROCFFT_PROFILE_COUNTER(last_child_exit);
    ROCFFT_PROFILE_COUNTER(child_signals);
    ROCFFT_PROFILE_COUNTER(last_child_signal);
    file << "\"store_errors\":" << s.store_errors.load() << "},\"files\":{";
#undef ROCFFT_PROFILE_COUNTER
    std::vector<std::string> paths{database, database + "-wal",
                                   database + "-shm", output};
    for (size_t i = 0; i < paths.size(); ++i) {
      if (i)
        file << ',';
      std::error_code error;
      auto size = std::filesystem::file_size(paths[i], error);
      file << json_quote(paths[i]) << ':';
      if (error)
        file << "null";
      else
        file << size;
    }
    file << "},\"properties\":{";
    {
      std::lock_guard<std::mutex> lock(s.detail_mutex);
      for (size_t i = 0; i < s.properties.size(); ++i) {
        if (i)
          file << ',';
        file << json_quote(s.properties[i].first) << ':'
             << json_quote(s.properties[i].second);
      }
      file << "},\"bounded_subprocess_errors\":[";
      for (size_t i = 0; i < s.failures.size(); ++i) {
        if (i)
          file << ',';
        file << json_quote(s.failures[i]);
      }
    }
    file << "],\"getrusage\":{\"self\":";
    write_usage(file, initial_self, cpu_usage(false));
    file << ",\"waited_children\":";
    write_usage(file, initial_children, cpu_usage(true));
    file << "}}\n";
    file.flush();
  }

public:
  Session(const std::string &architecture, const std::string &db,
          const std::string &out)
      : arch(architecture), database(db), output(out) {
    auto directory = std::getenv("ROCFFT_BUILD_PROFILE_DIR");
    if (!directory || !*directory)
      return;
    initial_self = cpu_usage(false);
    initial_children = cpu_usage(true);
    std::filesystem::create_directories(directory);
#ifndef _WIN32
    auto pid = std::to_string(getpid());
#else
    auto pid = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
    file.open(std::filesystem::path(directory) / (arch + "-" + pid + ".jsonl"));
    if (!file)
      throw std::runtime_error("cannot open requested rocFFT profile output");
    start = Clock::now();
    state().enabled.store(true);
    property("timers_overlap", "true; active occupancy and completed durations "
                               "are not additive wall time");
    property("getrusage_scope",
             "OS cumulative counters and session deltas; self includes "
             "reporter CPU; child CPU excludes escaped/unwaited descendants; "
             "child maxrss is a maximum, not a sum; final sample precedes its "
             "own report-write/exit overhead");
    snapshot("start");
    sampler = std::thread([this] {
      auto cpu_start = thread_cpu_ns();
      std::unique_lock<std::mutex> lock(stop_mutex);
      while (!stop_cv.wait_for(lock, std::chrono::seconds(1),
                               [this] { return stop; })) {
        sampler_cpu = thread_cpu_ns() - cpu_start;
        snapshot("sample");
      }
      sampler_cpu = thread_cpu_ns() - cpu_start;
    });
  }
  ~Session() {
    if (!file.is_open())
      return;
    {
      std::lock_guard<std::mutex> lock(stop_mutex);
      stop = true;
    }
    stop_cv.notify_one();
    sampler.join();
    snapshot("complete");
    state().enabled.store(false);
  }
};
} // namespace rocfft_build_profile
