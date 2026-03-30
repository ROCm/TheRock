/*
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * Minimal HIP vector-add used by the code-object-version backward-compatibility
 * test.  The program is intentionally self-contained (no external headers beyond
 * the HIP runtime) so it can live inside the test tree and compile against any
 * ROCm install without needing the rocm-systems submodule.
 */

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define CHECK_HIP(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,               \
                   hipGetErrorString(err), __FILE__, __LINE__);                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void vecadd(const float* A, const float* B, float* C, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

int main() {
  constexpr size_t N = 1024;
  constexpr size_t bytes = N * sizeof(float);

  float* h_A = static_cast<float*>(std::malloc(bytes));
  float* h_B = static_cast<float*>(std::malloc(bytes));
  float* h_C = static_cast<float*>(std::malloc(bytes));
  if (!h_A || !h_B || !h_C) {
    std::fprintf(stderr, "Host allocation failed\n");
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(i * 2);
  }

  float *d_A, *d_B, *d_C;
  CHECK_HIP(hipMalloc(&d_A, bytes));
  CHECK_HIP(hipMalloc(&d_B, bytes));
  CHECK_HIP(hipMalloc(&d_C, bytes));

  CHECK_HIP(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));

  constexpr unsigned threadsPerBlock = 256;
  unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  vecadd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
  CHECK_HIP(hipGetLastError());
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; ++i) {
    float expected = static_cast<float>(i) + static_cast<float>(i * 2);
    if (h_C[i] != expected) {
      std::fprintf(stderr, "Mismatch at %zu: got %f, expected %f\n", i, h_C[i],
                   expected);
      return EXIT_FAILURE;
    }
  }

  CHECK_HIP(hipFree(d_A));
  CHECK_HIP(hipFree(d_B));
  CHECK_HIP(hipFree(d_C));
  std::free(h_A);
  std::free(h_B);
  std::free(h_C);

  std::printf("PASSED!\n");
  return 0;
}
