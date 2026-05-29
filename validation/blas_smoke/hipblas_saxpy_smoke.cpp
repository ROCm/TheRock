#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void checkpoint(const char* what) {
  std::cerr << "[hipblas-smoke] " << what << "\n";
}

void checkHip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void checkHipblas(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << what << ": hipBLAS status " << static_cast<int>(status) << "\n";
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

  hipStream_t stream = nullptr;
  checkHip(hipStreamCreate(&stream), "hipStreamCreate");

  hipblasHandle_t handle = nullptr;
  checkHipblas(hipblasCreate(&handle), "hipblasCreate");
  checkHipblas(hipblasSetStream(handle, stream), "hipblasSetStream");
  checkHipblas(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST),
               "hipblasSetPointerMode");
  checkpoint("hipBLAS handle ready");

  constexpr int n = 64;
  constexpr float alpha = 2.0f;
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i + 1);
    y[i] = 100.0f + static_cast<float>(i);
  }
  const std::vector<float> y_initial = y;

  float* dx = nullptr;
  float* dy = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&dx), n * sizeof(float)), "hipMalloc x");
  checkHip(hipMalloc(reinterpret_cast<void**>(&dy), n * sizeof(float)), "hipMalloc y");
  checkHip(hipMemcpyAsync(dx, x.data(), n * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipMemcpyAsync H2D x");
  checkHip(hipMemcpyAsync(dy, y.data(), n * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipMemcpyAsync H2D y");
  checkpoint("device buffers initialized");

  checkHipblas(hipblasSaxpy(handle, n, &alpha, dx, 1, dy, 1), "hipblasSaxpy");
  checkHip(hipMemcpyAsync(y.data(), dy, n * sizeof(float), hipMemcpyDeviceToHost, stream),
           "hipMemcpyAsync D2H y");
  checkHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
  checkpoint("SAXPY synchronized");

  std::cout << "y[0..7]=";
  for (int i = 0; i < 8; ++i) {
    std::cout << (i == 0 ? "" : ",") << y[i];
  }
  std::cout << "\n";

  for (int i = 0; i < n; ++i) {
    const float expected = alpha * x[i] + y_initial[i];
    if (std::fabs(y[i] - expected) > 1e-4f) {
      std::cerr << "mismatch at " << i << ": got " << y[i]
                << ", expected " << expected << "\n";
      return 1;
    }
  }

  checkHip(hipFree(dy), "hipFree y");
  checkHip(hipFree(dx), "hipFree x");
  checkHipblas(hipblasDestroy(handle), "hipblasDestroy");
  checkHip(hipStreamDestroy(stream), "hipStreamDestroy");

  std::cout << "hipBLAS SAXPY smoke passed: " << y[0] << "\n";
  return 0;
}
