#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void checkpoint(const char* what) {
  std::cerr << "[hiprand-smoke] " << what << "\n";
}

void checkHip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void checkHiprand(hiprandStatus_t status, const char* what) {
  if (status != HIPRAND_STATUS_SUCCESS) {
    std::cerr << what << ": hipRAND status " << static_cast<int>(status) << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  checkpoint("start");

  checkHip(hipInit(0), "hipInit");
  checkHip(hipSetDevice(0), "hipSetDevice");
  checkpoint("hip initialized");

  hipDeviceProp_t prop{};
  checkHip(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::cout << "device=" << prop.name << " gfx=" << prop.gcnArchName << "\n";

  constexpr size_t n = 256;
  float* device_data = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&device_data), n * sizeof(float)), "hipMalloc");

  hiprandGenerator_t generator = nullptr;
  checkHiprand(hiprandCreateGenerator(&generator, HIPRAND_RNG_PSEUDO_PHILOX4_32_10),
               "hiprandCreateGenerator");
  checkHiprand(hiprandSetPseudoRandomGeneratorSeed(generator, 0x5eed), "hiprandSetSeed");
  checkHiprand(hiprandGenerateUniform(generator, device_data, n), "hiprandGenerateUniform");

  std::vector<float> host(n);
  checkHip(hipMemcpy(host.data(), device_data, n * sizeof(float), hipMemcpyDeviceToHost),
           "hipMemcpy D2H");
  checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  checkpoint("generation synchronized");

  size_t in_range = 0;
  float min_value = host[0];
  float max_value = host[0];
  double sum = 0.0;
  for (float value : host) {
    if (std::isfinite(value) && value > 0.0f && value <= 1.0f) ++in_range;
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    sum += value;
  }

  checkHiprand(hiprandDestroyGenerator(generator), "hiprandDestroyGenerator");
  checkHip(hipFree(device_data), "hipFree");

  if (in_range != n) {
    std::cerr << "range check failed: " << in_range << "/" << n << " values in (0,1]\n";
    return 1;
  }
  if (min_value == max_value) {
    std::cerr << "random output is constant: " << min_value << "\n";
    return 1;
  }

  std::cout << "hipRAND uniform smoke passed: min=" << min_value
            << " max=" << max_value << " mean=" << (sum / n) << "\n";
  return 0;
}
