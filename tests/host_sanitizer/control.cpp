// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <atomic>
#include <cstring>
#include <thread>

extern "C" void c_probe(int *);
extern "C" void cxx_probe(int *);
extern "C" void hip_probe(int *);

int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  auto probe = std::strcmp(argv[1], "c") == 0     ? c_probe
               : std::strcmp(argv[1], "cxx") == 0 ? cxx_probe
                                                  : hip_probe;
  if (std::strcmp(argv[2], "use-after-free") == 0) {
    auto *value = new int{0};
    delete value;
    probe(value);
    return 0;
  }

  int values[2] = {0, 0};
  const bool race = std::strcmp(argv[2], "race") == 0;
  std::atomic<int> ready{0};
  auto work = [&](int index) {
    ready.fetch_add(1);
    while (ready.load() != 2)
      std::this_thread::yield();
    for (int i = 0; i < 10000; ++i)
      probe(&values[race ? 0 : index]);
  };
  std::thread first(work, 0), second(work, 1);
  first.join();
  second.join();
  return race || (values[0] == 10000 && values[1] == 10000) ? 0 : 3;
}
