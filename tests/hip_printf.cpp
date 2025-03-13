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
  hipMalloc(&d_buf, size * sizeof(int));
  hipLaunchKernelGGL(squares, gridsize, blocksize, 0, 0, d_buf);
  hipDeviceSynchronize();
}
