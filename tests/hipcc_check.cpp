#include <cstdio>
#include <hip/hip_runtime.h>
__global__ void squares(int *buf) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Thread %#04x is writing %4d\n", i, i * i);
  buf[i] = i * i;
}
int main() {
  constexpr int gridsize = 1;
  constexpr int blocksize = 64;
  constexpr int size = gridsize * blocksize;
  int *d_buf;
  hipHostMalloc(&d_buf, size * sizeof(int));
  hipLaunchKernelGGL(squares, gridsize, blocksize, 0, 0, d_buf);
  hipDeviceSynchronize();

  // Check results.
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

  return 0;
}
