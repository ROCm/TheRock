// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Minimal HIP kernel WITHOUT printf, to isolate whether the gfx1150
// sanity-test hang (TheRock#3199) is caused by printf buffer handling
// or by kernel dispatch itself.

#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void squares_no_printf(int *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  buf[i] = i * i;
}

int main() {
  constexpr int gridsize = 1;
  constexpr int blocksize = 64;
  constexpr int size = gridsize * blocksize;
  int *d_buf;
  fprintf(stderr, "hip_simple_check: hipHostMalloc\n");
  hipHostMalloc(&d_buf, size * sizeof(int));
  fprintf(stderr, "hip_simple_check: hipLaunchKernelGGL\n");
  hipLaunchKernelGGL(squares_no_printf, gridsize, blocksize, 0, 0, d_buf);
  fprintf(stderr, "hip_simple_check: hipDeviceSynchronize\n");
  hipDeviceSynchronize();
  fprintf(stderr, "hip_simple_check: checking results\n");

  int mismatches_count = 0;
  for (int i = 0; i < size; ++i) {
    int square = i * i;
    if (d_buf[i] != square) {
      fprintf(stderr,
              "Element at index %d expected value %d, actual value: %d\n", i,
              square, d_buf[i]);
      ++mismatches_count;
    }
  }
  if (mismatches_count > 0) {
    fprintf(stderr, "There were %d mismatches\n", mismatches_count);
    return 1;
  }

  fprintf(stderr, "hip_simple_check: PASSED\n");
  return 0;
}
