// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <hip/hip_runtime.h>

// Simple vector addition kernel: C = A + B
__global__ void vector_add(const int *A, const int *B, int *C, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  constexpr int N = 256;
  constexpr int blocksize = 64;
  constexpr int gridsize = (N + blocksize - 1) / blocksize;

  // Host arrays
  int h_A[N], h_B[N], h_C[N];

  // Initialize input arrays
  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = i * 2;
  }

  // Device arrays
  int *d_A, *d_B, *d_C;
  hipMalloc(&d_A, N * sizeof(int));
  hipMalloc(&d_B, N * sizeof(int));
  hipMalloc(&d_C, N * sizeof(int));

  // Copy input data to device
  hipMemcpy(d_A, h_A, N * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B, N * sizeof(int), hipMemcpyHostToDevice);

  // Launch kernel
  hipLaunchKernelGGL(vector_add, gridsize, blocksize, 0, 0, d_A, d_B, d_C, N);
  hipDeviceSynchronize();

  // Copy result back to host
  hipMemcpy(h_C, d_C, N * sizeof(int), hipMemcpyDeviceToHost);

  // Check results
  int mismatches_count = 0;
  for (int i = 0; i < N; ++i) {
    int expected = h_A[i] + h_B[i];
    if (h_C[i] != expected) {
      fprintf(stderr,
              "Element at index %d expected value %d, actual value: %d\n", i,
              expected, h_C[i]);
      ++mismatches_count;
    }
  }

  // Cleanup
  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);

  if (mismatches_count > 0) {
    fprintf(stderr, "There were %d mismatches\n", mismatches_count);
    return 1;
  }

  return 0;
}
