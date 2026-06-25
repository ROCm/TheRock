// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {

__device__ constexpr index_t get_warp_size()
{
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX9__)
    return 64;
#else
    return 32;
#endif
#else
    return 64;
#endif
}

inline __host__ index_t get_warp_size()
{
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
    int device  = 0;
    int result  = 0;
    auto status = hipGetDevice(&device);
    if(status == hipSuccess)
    {
        status = hipDeviceGetAttribute(&result, hipDeviceAttributeWarpSize, device);
        if(status == hipSuccess)
        {
            return result;
        }
    }
#endif
    return 64;
}

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

__device__ index_t get_warp_local_1d_id() { return threadIdx.x / get_warp_size(); }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

__device__ index_t get_block_size() { return blockDim.x; }

} // namespace ck
