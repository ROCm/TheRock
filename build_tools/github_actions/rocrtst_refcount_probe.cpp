// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <chrono>
#include <climits>
#include <cstdio>
#include <hsa/hsa.h>

int main() {
  constexpr unsigned iterations = 100000;
  auto require_success = [](hsa_status_t status) {
    if (status != HSA_STATUS_SUCCESS) {
      std::fprintf(stderr, "HSA reference calibration failed: status=%u\n", unsigned(status));
      return false;
    }
    return true;
  };
  if (!require_success(hsa_init())) return 1;
  auto begin = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < iterations; ++i)
    if (!require_success(hsa_init())) return 1;
  auto middle = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < iterations; ++i)
    if (!require_success(hsa_shut_down())) return 1;
  auto end = std::chrono::steady_clock::now();
  if (!require_success(hsa_shut_down())) return 1;
  double init = std::chrono::duration<double>(middle - begin).count();
  double shutdown = std::chrono::duration<double>(end - middle).count();
  std::printf("Actual preloaded HSA calibration: %u init/shutdown references; "
              "init=%.6fs shutdown=%.6fs, %.1fns/pair; "
              "INT32_MAX linear estimate=%.1fs (not a completion guarantee)\n",
              iterations, init, shutdown, (init + shutdown) * 1e9 / iterations,
              (init + shutdown) * INT_MAX / iterations);
  return 0;
}
