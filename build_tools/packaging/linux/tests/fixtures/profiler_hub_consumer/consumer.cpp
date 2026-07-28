// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Trivial downstream consumer of the installed profiler-hub package. Links
// against libprofiler-hub.so and calls a real, side-effect-free API entry
// point (storage_t::get_storage_version()) so that both link-time symbol
// resolution and runtime loader NEEDED-dependency resolution are exercised.
// A missing runtime dependency surfaces here as a loader failure, not just a
// link-time pass.

#include <profiler-hub/storage.hpp>

#include <cstdio>

int main() {
  profiler_hub::storage_t storage{":memory:",
                                  "profiler_hub_consumer_smoke_test"};
  profiler_hub::version_t version = storage.get_storage_version();

  std::printf("profiler-hub storage version: %u.%u.%u\n", version.major,
              version.minor, version.patch);
  return 0;
}
