/**
 * vector_add_kernel.hip
 *
 * Simple vector addition kernel.
 *
 * Compile to a code object (works on ANY platform — the GPU ISA is the same):
 *   amdclang++ --offload-arch=gfx942 --cuda-device-only -o vector_add.co -x hip vector_add_kernel.hip
 *
 * Or on a Linux machine with ROCm:
 *   hipcc --genco --offload-arch=gfx942 -o vector_add.co vector_add_kernel.hip
 */

#include <hip/hip_runtime.h>

extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
