#include <cstdint>
#include <string>

#include <hip/hip_runtime_api.h>

#include <iostream>

int main(int argc, char **argv) {
  int device_ordinal = -1;
  if (argc > 1) {
    try {
      device_ordinal = std::stoi(argv[1]);
    } catch (std::exception &e) {
      std::cerr << "could not parse device ordinal from command line: "
                << e.what() << "\n";
      return 1;
    }
  }

  int runtime_version;
  auto err = hipRuntimeGetVersion(&runtime_version);
  if (err != hipSuccess) {
    std::cerr << "Error getting runtime version: " << err << "\n";
    return 2;
  }
  std::cout << "HIP runtime version: " << runtime_version << "\n";
  err = hipInit(0);
  if (device_ordinal < 0) {
    // Not testing actual device - just linkage and ability to run at all.
    std::cout << "Not testing on GPU device (no device ordinal passed)\n";
    return 0;
  }
  if (err != hipSuccess) {
    std::cerr << "Error initializing HIP: " << err << " (ignored)\n";
    return 3;
  }

  // Get the device.
  hipDevice_t device;
  err = hipDeviceGet(&device, device_ordinal);
  if (err != hipSuccess) {
    std::cerr << "Error getting device ordinal " << device_ordinal << "\n";
    return 4;
  }

  // Get device name.
  char device_name[80];
  err = hipDeviceGetName(device_name, sizeof(device_name) - 1, device);
  if (err != hipSuccess) {
    std::cerr << "Error getting device name \n";
    return 5;
  }
  std::cout << "Device name: " << device_name << "\n";

  // Get device memory.
  size_t memory_size;
  err = hipDeviceTotalMem(&memory_size, device);
  if (err != hipSuccess) {
    std::cerr << "Error getting device memory\n";
    return 6;
  }
  std::cout << "Device memory: " << (memory_size / 1024 / 1024) << " MiB\n";

  return 0;
}
