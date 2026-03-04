#include <hip/hip_runtime.h>
#include <cstdint>

// Compute-bound kernel: each thread does `iters` iterations of arithmetic.
// Output: out[idx] = final computed value (prevents compiler from optimizing away).
// Use large `iters` (e.g. 100000+) to keep CUs busy for measurable time.
extern "C" __global__ void busy_kernel(uint32_t* out, uint32_t seed, uint32_t iters) {
    int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    uint32_t val = seed + (uint32_t)idx;
    for (uint32_t i = 0; i < iters; i++) {
        val = val * 2654435761u + 1;  // Knuth multiplicative hash
    }
    out[idx] = val;
}

// Reduction-style kernel: each thread reads N elements and accumulates.
// in: input array of size num_elements
// out: output array of size (num_elements / elements_per_thread)
// Each thread reduces `elements_per_thread` consecutive values.
extern "C" __global__ void reduce_kernel(
    const uint32_t* in, uint32_t* out,
    uint32_t num_elements, uint32_t elements_per_thread
) {
    int idx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    uint32_t start = (uint32_t)idx * elements_per_thread;
    uint32_t sum = 0;
    for (uint32_t i = 0; i < elements_per_thread && (start + i) < num_elements; i++) {
        sum += in[start + i];
    }
    out[idx] = sum;
}

// Compile with:
//   hipcc --genco -o compute_kernels.bundle --offload-arch=gfx942 compute_kernels.hip
//   clang-offload-bundler --type=o --input=compute_kernels.bundle \
//     --output=compute_kernels_gfx942.co \
//     --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --unbundle
