#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

namespace {

void checkpoint(const char* what) {
  std::cerr << "[hipfft-smoke] " << what << "\n";
}

void checkHip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::_Exit(1);
  }
}

void checkHipfft(hipfftResult status, const char* what) {
  if (status != HIPFFT_SUCCESS) {
    std::cerr << what << ": hipFFT status " << static_cast<int>(status) << "\n";
    std::_Exit(1);
  }
}

}  // namespace

int run() {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  checkpoint("start");

  checkHip(hipInit(0), "hipInit");
  checkHip(hipSetDevice(0), "hipSetDevice");
  checkpoint("hip initialized");

  hipDeviceProp_t prop{};
  checkHip(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::cout << "device=" << prop.name << " gfx=" << prop.gcnArchName << "\n";

  constexpr int n = 8;
  std::vector<hipfftComplex> host(n);
  host[0].x = 1.0f;
  host[0].y = 0.0f;
  for (int i = 1; i < n; ++i) {
    host[i].x = 0.0f;
    host[i].y = 0.0f;
  }

  hipfftComplex* device_data = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&device_data), host.size() * sizeof(hipfftComplex)),
           "hipMalloc");
  checkHip(hipMemcpy(device_data, host.data(), host.size() * sizeof(hipfftComplex),
                     hipMemcpyHostToDevice),
           "hipMemcpy H2D");

  hipfftHandle plan = 0;
  checkHipfft(hipfftPlan1d(&plan, n, HIPFFT_C2C, 1), "hipfftPlan1d");
  checkpoint("plan ready");
  checkHipfft(hipfftExecC2C(plan, device_data, device_data, HIPFFT_FORWARD), "hipfftExecC2C");
  checkHip(hipMemcpy(host.data(), device_data, host.size() * sizeof(hipfftComplex),
                     hipMemcpyDeviceToHost),
           "hipMemcpy D2H");
  checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  checkpoint("FFT synchronized");

  float max_abs_error = 0.0f;
  for (const auto& value : host) {
    max_abs_error = std::max(max_abs_error, std::fabs(value.x - 1.0f));
    max_abs_error = std::max(max_abs_error, std::fabs(value.y));
  }

  checkHipfft(hipfftDestroy(plan), "hipfftDestroy");
  checkHip(hipFree(device_data), "hipFree");

  if (max_abs_error > 1e-5f) {
    std::cerr << "FFT mismatch: max_abs_error=" << max_abs_error << "\n";
    for (int i = 0; i < n; ++i) {
      std::cerr << "  y[" << i << "]=" << host[i].x << "+" << host[i].y << "i\n";
    }
    return 1;
  }

  std::cout << "hipFFT C2C smoke passed: max_abs_error=" << max_abs_error << "\n";
  return 0;
}

int main() {
  try {
    return run();
  } catch (const std::exception& e) {
    std::cerr << "hipFFT smoke exception: " << e.what() << "\n";
    std::_Exit(1);
  } catch (...) {
    std::cerr << "hipFFT smoke unknown exception\n";
    std::_Exit(1);
  }
}
