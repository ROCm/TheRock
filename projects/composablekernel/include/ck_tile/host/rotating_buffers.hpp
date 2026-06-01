// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>

namespace ck_tile {

// RotatingMemWrapper: Prevents GPU data cache reuse during kernel benchmarking.
//
// Purpose:
//   When benchmarking a kernel repeatedly with the same input buffers, the GPU L2 cache
//   will serve data from cache (hot) instead of HBM (cold), leading to artificially fast
//   timing measurements. This wrapper rotates through multiple copies of buffers at different
//   memory addresses to force cache misses.
//
// How it works:
//   Constructor: Creates rotating_count copies of matrices A and B in GPU memory
//   Next():      Switches pointers to the next buffer copy (cycles through all copies)
//   Destructor:  Frees extra buffer copies and restores original pointers
//
// Combined with flush_icache(), this ensures realistic "cold cache" performance measurements.
template <typename ADataType, typename BDataType>
struct RotatingMemWrapper
{
    RotatingMemWrapper() = delete;
    RotatingMemWrapper(const void* a_ptr_,
                       const void* b_ptr_,
                       std::size_t rotating_count_hint,
                       std::size_t size_a_,
                       std::size_t size_b_)
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          rotating_count(rotating_count_hint),
          size_a(size_a_),
          size_b(size_b_)
    {
        // Store original buffer pointers as first entry
        p_a_grids.push_back(a_ptr);
        p_b_grids.push_back(b_ptr);

        // limit the rotating count to prevent oom
        const uint64_t footprint          = (size_a + size_b);
        const uint64_t max_rotating_count = (1ULL << 31) / footprint;
        rotating_count                    = std::min(rotating_count, max_rotating_count);

        // Create (rotating_count - 1) additional copies at different memory addresses
        for(size_t i = 1; i < rotating_count; i++)
        {
            {
                void* pADeviceBuf;
                HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&pADeviceBuf), size_a_));
                HIP_CHECK_ERROR(hipMemcpy(static_cast<void*>(pADeviceBuf), // target buffer
                                          const_cast<void*>(p_a_grids[0]), // source buffer
                                          size_a_,
                                          hipMemcpyDeviceToDevice));
                p_a_grids.push_back(pADeviceBuf);
            }

            {
                void* pBDeviceBuf;
                HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&pBDeviceBuf), size_b_));
                HIP_CHECK_ERROR(hipMemcpy(static_cast<void*>(pBDeviceBuf), // target buffer
                                          const_cast<void*>(p_b_grids[0]), // source buffer
                                          size_b_,
                                          hipMemcpyDeviceToDevice));
                p_b_grids.push_back(pBDeviceBuf);
            }
        }
    }
    // Rotate to the next buffer copy. Call this before each kernel run to use different
    // memory addresses, forcing the GPU to fetch data from HBM instead of cache.
    void Next()
    {
        if(rotating_count > 1)
        {
            std::size_t idx = iter++ % rotating_count; // Cycle through all buffer copies
            a_ptr           = p_a_grids[idx];
            b_ptr           = p_b_grids[idx];
        }
    }
    void Print()
    {
        std::cout << "RotatingMemWrapper: { size_a: " << size_a << ", size_b: " << size_b
                  << ", rotating_count: " << rotating_count << "}" << std::endl;
    }
    // Cleanup: Free all extra buffer copies (keeping original) and restore original pointers
    ~RotatingMemWrapper() noexcept
    {
        if(rotating_count > 1)
        {
            // Restore original buffer pointers
            a_ptr = p_a_grids[0];
            b_ptr = p_b_grids[0];

            // Free extra buffer copies (index 0 is the original, don't free it)
            for(size_t i = 1; i < rotating_count; i++)
            {
                ck_tile::hip_check_error(hipFree(const_cast<void*>(p_a_grids[i])));
                ck_tile::hip_check_error(hipFree(const_cast<void*>(p_b_grids[i])));
            }
        }
    }

    private:
    const void* a_ptr;
    const void* b_ptr;
    std::size_t iter           = 0;
    std::size_t rotating_count = 1;
    std::size_t size_a         = 0;
    std::size_t size_b         = 0;
    std::vector<const void*> p_a_grids;
    std::vector<const void*> p_b_grids;
};
inline void flush_icache()
{
    hipDeviceProp_t deviceProps;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&deviceProps, 0));

    // Over-provision blocks to ensure all CUs execute the flush instruction.
    // With imperfect scheduling, launching exactly 1 block per CU doesn't guarantee coverage.
    // 60x over-provisioning provides statistical certainty that every CU gets at least one block.
    constexpr int32_t blocks_per_cu = 60;
    int32_t gpu_block3              = deviceProps.multiProcessorCount * blocks_per_cu;

    ck_tile::flush_cache<<<dim3(gpu_block3), dim3(64), 0, nullptr>>>();
    HIP_CHECK_ERROR(hipGetLastError());
}
} // namespace ck_tile
