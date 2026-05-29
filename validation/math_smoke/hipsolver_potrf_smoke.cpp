#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void checkpoint(const char* what) {
  std::cerr << "[hipsolver-potrf-smoke] " << what << "\n";
}

void checkHip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void checkHipsolver(hipsolverStatus_t status, const char* what) {
  if (status != HIPSOLVER_STATUS_SUCCESS) {
    std::cerr << what << ": hipSOLVER status " << static_cast<int>(status) << "\n";
    std::exit(1);
  }
}

int colMajor(int row, int col, int leading_dim) {
  return row + col * leading_dim;
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

  hipsolverHandle_t handle = nullptr;
  checkHipsolver(hipsolverDnCreate(&handle), "hipsolverDnCreate");
  checkHipsolver(hipsolverDnSetStream(handle, stream), "hipsolverDnSetStream");
  checkpoint("hipSOLVER handle ready");

  constexpr int n = 2;
  constexpr int lda = n;
  std::vector<float> a(lda * n);
  a[colMajor(0, 0, lda)] = 4.0f;
  a[colMajor(1, 0, lda)] = 1.0f;
  a[colMajor(0, 1, lda)] = 1.0f;
  a[colMajor(1, 1, lda)] = 3.0f;

  float* da = nullptr;
  int* dinfo = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&da), a.size() * sizeof(float)), "hipMalloc A");
  checkHip(hipMalloc(reinterpret_cast<void**>(&dinfo), sizeof(int)), "hipMalloc info");
  checkHip(hipMemcpyAsync(da, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipMemcpyAsync H2D A");

  int lwork = 0;
  checkHipsolver(hipsolverDnSpotrf_bufferSize(handle,
                                              HIPSOLVER_FILL_MODE_LOWER,
                                              n,
                                              da,
                                              lda,
                                              &lwork),
                 "hipsolverDnSpotrf_bufferSize");
  checkpoint("workspace sized");

  float* dwork = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&dwork), std::max(lwork, 1) * sizeof(float)),
           "hipMalloc work");
  checkHipsolver(hipsolverDnSpotrf(handle,
                                   HIPSOLVER_FILL_MODE_LOWER,
                                   n,
                                   da,
                                   lda,
                                   dwork,
                                   lwork,
                                   dinfo),
                 "hipsolverDnSpotrf");

  int info = -1;
  checkHip(hipMemcpyAsync(a.data(), da, a.size() * sizeof(float), hipMemcpyDeviceToHost, stream),
           "hipMemcpyAsync D2H A");
  checkHip(hipMemcpyAsync(&info, dinfo, sizeof(int), hipMemcpyDeviceToHost, stream),
           "hipMemcpyAsync D2H info");
  checkHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
  checkpoint("POTRF synchronized");

  if (info != 0) {
    std::cerr << "hipsolverDnSpotrf returned devInfo=" << info << "\n";
    return 1;
  }

  const float expected_l00 = 2.0f;
  const float expected_l10 = 0.5f;
  const float expected_l11 = std::sqrt(2.75f);
  float max_abs_error = 0.0f;
  max_abs_error = std::max(max_abs_error, std::fabs(a[colMajor(0, 0, lda)] - expected_l00));
  max_abs_error = std::max(max_abs_error, std::fabs(a[colMajor(1, 0, lda)] - expected_l10));
  max_abs_error = std::max(max_abs_error, std::fabs(a[colMajor(1, 1, lda)] - expected_l11));
  if (max_abs_error > 1e-4f) {
    std::cerr << "unexpected Cholesky result: L00=" << a[colMajor(0, 0, lda)]
              << " L10=" << a[colMajor(1, 0, lda)]
              << " L11=" << a[colMajor(1, 1, lda)]
              << " max_abs_error=" << max_abs_error << "\n";
    return 1;
  }

  checkHip(hipFree(dwork), "hipFree work");
  checkHip(hipFree(dinfo), "hipFree info");
  checkHip(hipFree(da), "hipFree A");
  checkHipsolver(hipsolverDnDestroy(handle), "hipsolverDnDestroy");
  checkHip(hipStreamDestroy(stream), "hipStreamDestroy");

  std::cout << "hipSOLVER POTRF smoke passed: max_abs_error=" << max_abs_error << "\n";
  return 0;
}
