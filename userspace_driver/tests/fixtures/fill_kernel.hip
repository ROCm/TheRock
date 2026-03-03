#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void fill_kernel(uint32_t* out, uint32_t val) {
    int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    out[idx] = val;
}

// Compile with:
//   hipcc --genco -o fill_kernel.bundle --offload-arch=gfx942 fill_kernel.hip
//   clang-offload-bundler --type=o --input=fill_kernel.bundle \
//     --output=fill_kernel_gfx942.co \
//     --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --unbundle
