#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void checkpoint(const char* what) {
  std::cerr << "[hipblas-gemm-smoke] " << what << "\n";
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

  hipblasHandle_t handle = nullptr;
  checkHipblas(hipblasCreate(&handle), "hipblasCreate");
  checkHipblas(hipblasSetStream(handle, stream), "hipblasSetStream");
  checkHipblas(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST),
               "hipblasSetPointerMode");
  checkpoint("hipBLAS handle ready");

  constexpr int m = 8;
  constexpr int n = 7;
  constexpr int k = 6;
  constexpr int lda = m;
  constexpr int ldb = k;
  constexpr int ldc = m;
  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  std::vector<float> a(lda * k);
  std::vector<float> b(ldb * n);
  std::vector<float> c(ldc * n, -1.0f);
  std::vector<float> expected(ldc * n, 0.0f);

  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < m; ++row) {
      a[colMajor(row, col, lda)] = static_cast<float>((row + 1) + 0.25f * (col + 1));
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < k; ++row) {
      b[colMajor(row, col, ldb)] = static_cast<float>((col + 1) - 0.5f * (row + 1));
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < m; ++row) {
      float sum = 0.0f;
      for (int p = 0; p < k; ++p) {
        sum += a[colMajor(row, p, lda)] * b[colMajor(p, col, ldb)];
      }
      expected[colMajor(row, col, ldc)] = sum;
    }
  }

  float* da = nullptr;
  float* db = nullptr;
  float* dc = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&da), a.size() * sizeof(float)), "hipMalloc A");
  checkpoint("allocated A");
  checkHip(hipMalloc(reinterpret_cast<void**>(&db), b.size() * sizeof(float)), "hipMalloc B");
  checkpoint("allocated B");
  checkHip(hipMalloc(reinterpret_cast<void**>(&dc), c.size() * sizeof(float)), "hipMalloc C");
  checkpoint("allocated C");
  checkHip(hipMemcpyAsync(da, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipMemcpyAsync H2D A");
  checkpoint("copied A");
  checkHip(hipMemcpyAsync(db, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipMemcpyAsync H2D B");
  checkpoint("copied B");
  checkHip(hipMemcpyAsync(dc, c.data(), c.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipMemcpyAsync H2D C");
  checkpoint("device buffers initialized");

  checkHipblas(hipblasSgemm(handle,
                            HIPBLAS_OP_N,
                            HIPBLAS_OP_N,
                            m,
                            n,
                            k,
                            &alpha,
                            da,
                            lda,
                            db,
                            ldb,
                            &beta,
                            dc,
                            ldc),
               "hipblasSgemm");
  checkHip(hipMemcpyAsync(c.data(), dc, c.size() * sizeof(float), hipMemcpyDeviceToHost, stream),
           "hipMemcpyAsync D2H C");
  checkHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
  checkpoint("SGEMM synchronized");

  float max_abs_error = 0.0f;
  for (size_t i = 0; i < c.size(); ++i) {
    max_abs_error = std::max(max_abs_error, std::fabs(c[i] - expected[i]));
    if (std::fabs(c[i] - expected[i]) > 1e-3f) {
      std::cerr << "mismatch at " << i << ": got " << c[i]
                << ", expected " << expected[i] << "\n";
      return 1;
    }
  }

  checkHip(hipFree(dc), "hipFree C");
  checkHip(hipFree(db), "hipFree B");
  checkHip(hipFree(da), "hipFree A");
  checkHipblas(hipblasDestroy(handle), "hipblasDestroy");
  checkHip(hipStreamDestroy(stream), "hipStreamDestroy");

  std::cout << "C[0]=" << c[0] << " C[last]=" << c.back()
            << " max_abs_error=" << max_abs_error << "\n";
  std::cout << "hipBLAS SGEMM smoke passed\n";
  return 0;
}
