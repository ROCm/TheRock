// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace ck {
struct workgroup_barrier
{
    __device__ workgroup_barrier(uint32_t* ptr) : base_ptr(ptr) {}

    __device__ uint32_t ld(uint32_t offset)
    {
#if 0
        float d = llvm_amdgcn_raw_buffer_load_fp32(
                        amdgcn_make_buffer_resource(base_ptr),
                        0,
                        offset,
                        AMDGCN_BUFFER_GLC);
        union cvt {
            float f32;
            uint32_t u32;
        };
        cvt x;
        x.f32 = d;
        return x.u32;
#endif
        return __atomic_load_n(base_ptr + offset, __ATOMIC_RELAXED);
    }

    __device__ void wait_eq(uint32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) != value) {}
        }
        __syncthreads();
    }

    __device__ void wait_lt(uint32_t offset, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(ld(offset) < value) {}
        }
        __syncthreads();
    }

    __device__ void wait_set(uint32_t offset, uint32_t compare, uint32_t value)
    {
        if(threadIdx.x == 0)
        {
            while(atomicCAS(base_ptr + offset, compare, value) != compare) {}
        }
        __syncthreads();
    }

    // enter critical zoon, assume buffer is zero when launch kernel
    __device__ void aquire(uint32_t offset) { wait_set(offset, 0, 1); }

    // exit critical zoon, assume buffer is zero when launch kernel
    __device__ void release(uint32_t offset) { wait_set(offset, 1, 0); }

    __device__ void inc(uint32_t offset)
    {
        __syncthreads();
        if(threadIdx.x == 0)
        {
            atomicAdd(base_ptr + offset, 1);
        }
    }

    uint32_t* base_ptr;
};
} // namespace ck
