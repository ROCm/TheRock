#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void checkpoint(const char* what) {
  std::cerr << "[hipsparse-smoke] " << what << "\n";
}

void checkHip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void checkHipsparse(hipsparseStatus_t status, const char* what) {
  if (status != HIPSPARSE_STATUS_SUCCESS) {
    std::cerr << what << ": hipSPARSE status " << static_cast<int>(status) << "\n";
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
           "hipMalloc row_ptr");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_col_ind), col_ind.size() * sizeof(int)),
           "hipMalloc col_ind");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_values), values.size() * sizeof(float)),
           "hipMalloc values");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(float)), "hipMalloc x");
  checkHip(hipMalloc(reinterpret_cast<void**>(&d_y), y.size() * sizeof(float)), "hipMalloc y");

  checkHip(hipMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int),
                     hipMemcpyHostToDevice),
           "hipMemcpy row_ptr");
  checkHip(hipMemcpy(d_col_ind, col_ind.data(), col_ind.size() * sizeof(int),
                     hipMemcpyHostToDevice),
           "hipMemcpy col_ind");
  checkHip(hipMemcpy(d_values, values.data(), values.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "hipMemcpy values");
  checkHip(hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
           "hipMemcpy x");
  checkHip(hipMemcpy(d_y, y.data(), y.size() * sizeof(float), hipMemcpyHostToDevice),
           "hipMemcpy y");

  hipsparseHandle_t handle = nullptr;
  checkHipsparse(hipsparseCreate(&handle), "hipsparseCreate");

  hipsparseMatDescr_t mat = nullptr;
  checkHipsparse(hipsparseCreateMatDescr(&mat), "hipsparseCreateMatDescr");
  checkHipsparse(hipsparseSetMatType(mat, HIPSPARSE_MATRIX_TYPE_GENERAL), "hipsparseSetMatType");
  checkHipsparse(hipsparseSetMatIndexBase(mat, HIPSPARSE_INDEX_BASE_ZERO),
                 "hipsparseSetMatIndexBase");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  checkHipsparse(hipsparseScsrmv(handle,
                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                 rows,
                                 cols,
                                 nnz,
                                 &alpha,
                                 mat,
                                 d_values,
                                 d_row_ptr,
                                 d_col_ind,
                                 d_x,
                                 &beta,
                                 d_y),
                 "hipsparseScsrmv");

  checkHip(hipMemcpy(y.data(), d_y, y.size() * sizeof(float), hipMemcpyDeviceToHost),
           "hipMemcpy D2H");
  checkHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  checkpoint("SpMV synchronized");

  float max_abs_error = 0.0f;
  for (size_t i = 0; i < y.size(); ++i) {
    max_abs_error = std::max(max_abs_error, std::fabs(y[i] - expected[i]));
  }

  checkHipsparse(hipsparseDestroyMatDescr(mat), "hipsparseDestroyMatDescr");
  checkHipsparse(hipsparseDestroy(handle), "hipsparseDestroy");
  checkHip(hipFree(d_row_ptr), "hipFree row_ptr");
  checkHip(hipFree(d_col_ind), "hipFree col_ind");
  checkHip(hipFree(d_values), "hipFree values");
  checkHip(hipFree(d_x), "hipFree x");
  checkHip(hipFree(d_y), "hipFree y");

  if (max_abs_error > 1e-5f) {
    std::cerr << "SpMV mismatch: max_abs_error=" << max_abs_error << "\n";
    for (size_t i = 0; i < y.size(); ++i) {
      std::cerr << "  y[" << i << "]=" << y[i] << " expected=" << expected[i] << "\n";
    }
    return 1;
  }

  std::cout << "hipSPARSE SpMV smoke passed: max_abs_error=" << max_abs_error << "\n";
  return 0;
}
