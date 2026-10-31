// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocfft_profile.h"
#include <iostream>
#ifndef _WIN32
#include <sys/wait.h>
#endif
void rocfft_profile_fixture_other_tu();

int main() {
  namespace rp = rocfft_build_profile;
  long long stop_ns = -1;
  auto session = std::make_unique<rp::Session>("fixture", "missing.db",
                                               "missing-output.db");
  std::mutex mutex;
  std::unique_lock<std::mutex> owner(mutex);
  std::thread worker([&] {
    rp::Scope parent(rp::future_wait);
    rp::Lock lock(mutex, rp::get_mutex_wait, rp::get_mutex_hold);
    rp::Scope generated(rp::generate);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(1150));
  owner.unlock();
  worker.join();
  auto loop_start = rp::Clock::now();
  for (int i = 0; i < 50000; ++i) {
    rp::Scope scope(rp::store_query);
  }
  auto loop_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     rp::Clock::now() - loop_start)
                     .count();
  try {
    rp::Scope scope(rp::inprocess_compile);
    throw std::runtime_error("fixture exception");
  } catch (const std::exception &) {
    for (int i = 0; i < 10; ++i)
      rp::failure(std::string(200, 'x').c_str());
    rp::add(rp::state().fallbacks);
  }
  rp::property("fixture_property", "a\"b\nc");
#ifndef _WIN32
  auto child = fork();
  if (child == 0) {
    volatile unsigned long long value = 1;
    for (int i = 0; i < 10000000; ++i)
      value = value * 3 + 1;
    _exit(value == 0 ? 1 : 0);
  }
  int status = 0;
  if (child < 0 || waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status))
    throw std::runtime_error("fixture child failed");
#endif
  rocfft_profile_fixture_other_tu();
  rp::error_code(rp::state().spawn_errors, rp::state().last_spawn_errno, 11);
  auto stopped = rp::Clock::now();
  session.reset();
  stop_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                rp::Clock::now() - stopped)
                .count();
  std::cout << "{\"stop_ns\":" << stop_ns << ",\"scope_loop_ns\":" << loop_ns
            << "}\n";
}
