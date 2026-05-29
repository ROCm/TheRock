#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipfft/hipfft.h>
#include <hiprand/hiprand.h>
#include <hipsolver/hipsolver.h>
#include <hipsparse/hipsparse.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

namespace {

void checkpoint(const char* what) {
  std::cerr << "[rocm-combined-smoke] " << what << "\n";
}

void fail(const char* what, const char* detail) {
  std::cerr << what << ": " << detail << "\n";
  std::_Exit(1);
}

void checkHip(hipError_t status, const char* what) {
  if (status != hipSuccess) fail(what, hipGetErrorString(status));
}

void checkHipfft(hipfftResult status, const char* what) {
  if (status != HIPFFT_SUCCESS) {
    std::cerr << what << ": hipFFT status " << static_cast<int>(status) << "\n";
    std::_Exit(1);
  }
}

void checkHipsolver(hipsolverStatus_t status, const char* what) {
  if (status != HIPSOLVER_STATUS_SUCCESS) {
    std::cerr << what << ": hipSOLVER status " << static_cast<int>(status) << "\n";
    std::_Exit(1);
  }
}

void checkHipsparse(hipsparseStatus_t status, const char* what) {
  if (status != HIPSPARSE_STATUS_SUCCESS) {
    std::cerr << what << ": hipSPARSE status " << static_cast<int>(status) << "\n";
    std::_Exit(1);
  }
}

void checkHiprand(hiprandStatus_t status, const char* what) {
  if (status != HIPRAND_STATUS_SUCCESS) {
    std::cerr << what << ": hipRAND status " << static_cast<int>(status) << "\n";
    std::_Exit(1);
  }
}

void checkHipblas(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << what << ": hipBLAS status " << static_cast<int>(status) << "\n";
    std::_Exit(1);
  }
}

int colMajor(int row, int col, int leading_dim) {
  return row + col * leading_dim;
}

void runHipfft(hipStream_t stream) {
  checkpoint("hipFFT start");

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
           "hipFFT hipMalloc");
  checkHip(hipMemcpyAsync(device_data, host.data(), host.size() * sizeof(hipfftComplex),
                          hipMemcpyHostToDevice, stream),
           "hipFFT hipMemcpyAsync H2D");

  hipfftHandle plan = 0;
  checkHipfft(hipfftPlan1d(&plan, n, HIPFFT_C2C, 1), "hipfftPlan1d");
  checkHipfft(hipfftSetStream(plan, stream), "hipfftSetStream");
  checkpoint("hipFFT plan ready");
  checkHipfft(hipfftExecC2C(plan, device_data, device_data, HIPFFT_FORWARD), "hipfftExecC2C");
  checkHip(hipMemcpyAsync(host.data(), device_data, host.size() * sizeof(hipfftComplex),
                          hipMemcpyDeviceToHost, stream),
           "hipFFT hipMemcpyAsync D2H");
  checkHip(hipStreamSynchronize(stream), "hipFFT hipStreamSynchronize");

  float max_abs_error = 0.0f;
  for (const auto& value : host) {
    max_abs_error = std::max(max_abs_error, std::fabs(value.x - 1.0f));
    max_abs_error = std::max(max_abs_error, std::fabs(value.y));
  }

  checkHipfft(hipfftDestroy(plan), "hipfftDestroy");
  checkHip(hipFree(device_data), "hipFFT hipFree");

  if (max_abs_error > 1e-5f) {
    std::cerr << "hipFFT mismatch: max_abs_error=" << max_abs_error << "\n";
    std::_Exit(1);
  }
  std::cout << "hipFFT C2C smoke passed: max_abs_error=" << max_abs_error << "\n";
}

void runHipsolver(hipStream_t stream) {
  checkpoint("hipSOLVER start");

  hipsolverHandle_t handle = nullptr;
  checkHipsolver(hipsolverDnCreate(&handle), "hipsolverDnCreate");
  checkHipsolver(hipsolverDnSetStream(handle, stream), "hipsolverDnSetStream");

  constexpr int n = 2;
  constexpr int lda = n;
  std::vector<float> a(lda * n);
  a[colMajor(0, 0, lda)] = 4.0f;
  a[colMajor(1, 0, lda)] = 1.0f;
  a[colMajor(0, 1, lda)] = 1.0f;
  a[colMajor(1, 1, lda)] = 3.0f;

  float* da = nullptr;
  int* dinfo = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&da), a.size() * sizeof(float)),
           "hipSOLVER hipMalloc A");
  checkHip(hipMalloc(reinterpret_cast<void**>(&dinfo), sizeof(int)), "hipSOLVER hipMalloc info");
  checkHip(hipMemcpyAsync(da, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipSOLVER hipMemcpyAsync H2D A");

  int lwork = 0;
  checkHipsolver(hipsolverDnSpotrf_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, n, da, lda,
                                              &lwork),
                 "hipsolverDnSpotrf_bufferSize");

  float* dwork = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&dwork), std::max(lwork, 1) * sizeof(float)),
           "hipSOLVER hipMalloc work");
  checkHipsolver(hipsolverDnSpotrf(handle, HIPSOLVER_FILL_MODE_LOWER, n, da, lda, dwork,
                                   lwork, dinfo),
                 "hipsolverDnSpotrf");

  int info = -1;
  checkHip(hipMemcpyAsync(a.data(), da, a.size() * sizeof(float), hipMemcpyDeviceToHost, stream),
           "hipSOLVER hipMemcpyAsync D2H A");
  checkHip(hipMemcpyAsync(&info, dinfo, sizeof(int), hipMemcpyDeviceToHost, stream),
           "hipSOLVER hipMemcpyAsync D2H info");
  checkHip(hipStreamSynchronize(stream), "hipSOLVER hipStreamSynchronize");

  if (info != 0) {
    std::cerr << "hipsolverDnSpotrf returned devInfo=" << info << "\n";
    std::_Exit(1);
  }

  const float expected_l00 = 2.0f;
  const float expected_l10 = 0.5f;
  const float expected_l11 = std::sqrt(2.75f);
  float max_abs_error = 0.0f;
  max_abs_error = std::max(max_abs_error, std::fabs(a[colMajor(0, 0, lda)] - expected_l00));
  max_abs_error = std::max(max_abs_error, std::fabs(a[colMajor(1, 0, lda)] - expected_l10));
  max_abs_error = std::max(max_abs_error, std::fabs(a[colMajor(1, 1, lda)] - expected_l11));
  if (max_abs_error > 1e-4f) {
    std::cerr << "hipSOLVER mismatch: max_abs_error=" << max_abs_error << "\n";
    std::_Exit(1);
  }

  checkHip(hipFree(dwork), "hipSOLVER hipFree work");
  checkHip(hipFree(dinfo), "hipSOLVER hipFree info");
  checkHip(hipFree(da), "hipSOLVER hipFree A");
  checkHipsolver(hipsolverDnDestroy(handle), "hipsolverDnDestroy");
  std::cout << "hipSOLVER POTRF smoke passed: max_abs_error=" << max_abs_error << "\n";
}

void runHipsparse(hipStream_t stream) {
  checkpoint("hipSPARSE start");

  constexpr int rows = 4;
  constexpr int cols = 4;
  constexpr int nnz = 8;
  const std::vector<int> row_ptr = {0, 3, 5, 7, 8};
  const std::vector<int> col_ind = {0, 1, 3, 1, 2, 0, 2, 3};
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> x = {1.0f, 1.0f, 1.0f, 1.0f};
  const std::vector<float> expected = {6.0f, 9.0f, 13.0f, 8.0f};
  std::vector<float> y(rows, 0.0f);

  int* d_row_ptr = nullptr;
  int* d_col_ind = nullptr;
  float* d_values = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_row_ptr), row_ptr.size() * sizeof(int)),
           "hipSPARSE hipMalloc row_ptr");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_col_ind), col_ind.size() * sizeof(int)),
           "hipSPARSE hipMalloc col_ind");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_values), values.size() * sizeof(float)),
           "hipSPARSE hipMalloc values");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(float)),
           "hipSPARSE hipMalloc x");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_y), y.size() * sizeof(float)),
           "hipSPARSE hipMalloc y");

  checkHip(hipMemcpyAsync(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int),
                          hipMemcpyHostToDevice, stream),
           "hipSPARSE hipMemcpyAsync row_ptr");
  checkHip(hipMemcpyAsync(d_col_ind, col_ind.data(), col_ind.size() * sizeof(int),
                          hipMemcpyHostToDevice, stream),
           "hipSPARSE hipMemcpyAsync col_ind");
  checkHip(hipMemcpyAsync(d_values, values.data(), values.size() * sizeof(float),
                          hipMemcpyHostToDevice, stream),
           "hipSPARSE hipMemcpyAsync values");
  checkHip(hipMemcpyAsync(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipSPARSE hipMemcpyAsync x");
  checkHip(hipMemcpyAsync(d_y, y.data(), y.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipSPARSE hipMemcpyAsync y");

  hipsparseHandle_t handle = nullptr;
  checkHipsparse(hipsparseCreate(&handle), "hipsparseCreate");
  checkHipsparse(hipsparseSetStream(handle, stream), "hipsparseSetStream");

  hipsparseMatDescr_t mat = nullptr;
  checkHipsparse(hipsparseCreateMatDescr(&mat), "hipsparseCreateMatDescr");
  checkHipsparse(hipsparseSetMatType(mat, HIPSPARSE_MATRIX_TYPE_GENERAL), "hipsparseSetMatType");
  checkHipsparse(hipsparseSetMatIndexBase(mat, HIPSPARSE_INDEX_BASE_ZERO),
                 "hipsparseSetMatIndexBase");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkHipsparse(hipsparseScsrmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, rows, cols, nnz,
                                 &alpha, mat, d_values, d_row_ptr, d_col_ind, d_x, &beta, d_y),
                 "hipsparseScsrmv");
  checkHip(hipMemcpyAsync(y.data(), d_y, y.size() * sizeof(float), hipMemcpyDeviceToHost, stream),
           "hipSPARSE hipMemcpyAsync D2H");
  checkHip(hipStreamSynchronize(stream), "hipSPARSE hipStreamSynchronize");

  float max_abs_error = 0.0f;
  for (size_t i = 0; i < y.size(); ++i) {
    max_abs_error = std::max(max_abs_error, std::fabs(y[i] - expected[i]));
  }
  if (max_abs_error > 1e-5f) {
    std::cerr << "hipSPARSE mismatch: max_abs_error=" << max_abs_error << "\n";
    std::_Exit(1);
  }

  checkHipsparse(hipsparseDestroyMatDescr(mat), "hipsparseDestroyMatDescr");
  checkHipsparse(hipsparseDestroy(handle), "hipsparseDestroy");
  checkHip(hipFree(d_row_ptr), "hipSPARSE hipFree row_ptr");
  checkHip(hipFree(d_col_ind), "hipSPARSE hipFree col_ind");
  checkHip(hipFree(d_values), "hipSPARSE hipFree values");
  checkHip(hipFree(d_x), "hipSPARSE hipFree x");
  checkHip(hipFree(d_y), "hipSPARSE hipFree y");
  std::cout << "hipSPARSE SpMV smoke passed: max_abs_error=" << max_abs_error << "\n";
}

void runHiprand(hipStream_t stream) {
  checkpoint("hipRAND start");

  constexpr size_t n = 256;
  float* device_data = nullptr;
  checkHip(hipMalloc(reinterpret_cast<void**>(&device_data), n * sizeof(float)),
           "hipRAND hipMalloc");

  hiprandGenerator_t generator = nullptr;
  checkHiprand(hiprandCreateGenerator(&generator, HIPRAND_RNG_PSEUDO_PHILOX4_32_10),
               "hiprandCreateGenerator");
  checkHiprand(hiprandSetStream(generator, stream), "hiprandSetStream");
  checkHiprand(hiprandSetPseudoRandomGeneratorSeed(generator, 0x5eed), "hiprandSetSeed");
  checkHiprand(hiprandGenerateUniform(generator, device_data, n), "hiprandGenerateUniform");

  std::vector<float> host(n);
  checkHip(hipMemcpyAsync(host.data(), device_data, n * sizeof(float), hipMemcpyDeviceToHost,
                          stream),
           "hipRAND hipMemcpyAsync D2H");
  checkHip(hipStreamSynchronize(stream), "hipRAND hipStreamSynchronize");

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
  if (in_range != n || min_value == max_value) {
    std::cerr << "hipRAND output check failed: in_range=" << in_range
              << " min=" << min_value << " max=" << max_value << "\n";
    std::_Exit(1);
  }

  checkHiprand(hiprandDestroyGenerator(generator), "hiprandDestroyGenerator");
  checkHip(hipFree(device_data), "hipRAND hipFree");
  std::cout << "hipRAND uniform smoke passed: min=" << min_value
            << " max=" << max_value << " mean=" << (sum / n) << "\n";
}

void runHipblasSaxpy(hipblasHandle_t handle, hipStream_t stream) {
  checkpoint("hipBLAS SAXPY start");

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
  checkHip(hipMalloc(reinterpret_cast<void**>(&dx), n * sizeof(float)), "hipBLAS hipMalloc x");
  checkHip(hipMalloc(reinterpret_cast<void**>(&dy), n * sizeof(float)), "hipBLAS hipMalloc y");
  checkHip(hipMemcpyAsync(dx, x.data(), n * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipBLAS SAXPY hipMemcpyAsync H2D x");
  checkHip(hipMemcpyAsync(dy, y.data(), n * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipBLAS SAXPY hipMemcpyAsync H2D y");

  checkHipblas(hipblasSaxpy(handle, n, &alpha, dx, 1, dy, 1), "hipblasSaxpy");
  checkHip(hipMemcpyAsync(y.data(), dy, n * sizeof(float), hipMemcpyDeviceToHost, stream),
           "hipBLAS SAXPY hipMemcpyAsync D2H y");
  checkHip(hipStreamSynchronize(stream), "hipBLAS SAXPY hipStreamSynchronize");

  for (int i = 0; i < n; ++i) {
    const float expected = alpha * x[i] + y_initial[i];
    if (std::fabs(y[i] - expected) > 1e-4f) {
      std::cerr << "hipBLAS SAXPY mismatch at " << i << ": got " << y[i]
                << ", expected " << expected << "\n";
      std::_Exit(1);
    }
  }

  checkHip(hipFree(dy), "hipBLAS SAXPY hipFree y");
  checkHip(hipFree(dx), "hipBLAS SAXPY hipFree x");
  std::cout << "hipBLAS SAXPY smoke passed: " << y[0] << "\n";
}

void runHipblasGemm(hipblasHandle_t handle, hipStream_t stream) {
  checkpoint("hipBLAS SGEMM start");

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
  checkHip(hipMalloc(reinterpret_cast<void**>(&da), a.size() * sizeof(float)),
           "hipBLAS SGEMM hipMalloc A");
  checkHip(hipMalloc(reinterpret_cast<void**>(&db), b.size() * sizeof(float)),
           "hipBLAS SGEMM hipMalloc B");
  checkHip(hipMalloc(reinterpret_cast<void**>(&dc), c.size() * sizeof(float)),
           "hipBLAS SGEMM hipMalloc C");
  checkHip(hipMemcpyAsync(da, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipBLAS SGEMM hipMemcpyAsync H2D A");
  checkHip(hipMemcpyAsync(db, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipBLAS SGEMM hipMemcpyAsync H2D B");
  checkHip(hipMemcpyAsync(dc, c.data(), c.size() * sizeof(float), hipMemcpyHostToDevice, stream),
           "hipBLAS SGEMM hipMemcpyAsync H2D C");

  checkHipblas(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, da, lda,
                            db, ldb, &beta, dc, ldc),
               "hipblasSgemm");
  checkHip(hipMemcpyAsync(c.data(), dc, c.size() * sizeof(float), hipMemcpyDeviceToHost, stream),
           "hipBLAS SGEMM hipMemcpyAsync D2H C");
  checkHip(hipStreamSynchronize(stream), "hipBLAS SGEMM hipStreamSynchronize");

  float max_abs_error = 0.0f;
  for (size_t i = 0; i < c.size(); ++i) {
    max_abs_error = std::max(max_abs_error, std::fabs(c[i] - expected[i]));
  }
  if (max_abs_error > 1e-3f) {
    std::cerr << "hipBLAS SGEMM mismatch: max_abs_error=" << max_abs_error << "\n";
    std::_Exit(1);
  }

  checkHip(hipFree(dc), "hipBLAS SGEMM hipFree C");
  checkHip(hipFree(db), "hipBLAS SGEMM hipFree B");
  checkHip(hipFree(da), "hipBLAS SGEMM hipFree A");
  std::cout << "hipBLAS SGEMM smoke passed: max_abs_error=" << max_abs_error << "\n";
}

void runHipblas(hipStream_t stream) {
  hipblasHandle_t handle = nullptr;
  checkHipblas(hipblasCreate(&handle), "hipblasCreate");
  checkHipblas(hipblasSetStream(handle, stream), "hipblasSetStream");
  checkHipblas(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST),
               "hipblasSetPointerMode");
  runHipblasSaxpy(handle, stream);
  runHipblasGemm(handle, stream);
  checkHipblas(hipblasDestroy(handle), "hipblasDestroy");
}

int run() {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);

  checkHip(hipInit(0), "hipInit");
  checkHip(hipSetDevice(0), "hipSetDevice");

  hipDeviceProp_t prop{};
  checkHip(hipGetDeviceProperties(&prop, 0), "hipGetDeviceProperties");
  std::cout << "device=" << prop.name << " gfx=" << prop.gcnArchName << "\n";

  hipStream_t stream = nullptr;
  checkHip(hipStreamCreate(&stream), "hipStreamCreate");

  runHipfft(stream);
  runHipsolver(stream);
  runHipsparse(stream);
  runHiprand(stream);
  runHipblas(stream);

  checkHip(hipStreamDestroy(stream), "hipStreamDestroy");
  std::cout << "ROCm combined stack smoke passed\n";
  return 0;
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& e) {
    std::cerr << "ROCm combined smoke exception: " << e.what() << "\n";
    std::_Exit(1);
  } catch (...) {
    std::cerr << "ROCm combined smoke unknown exception\n";
    std::_Exit(1);
  }
}
