// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck/host_utility/hip_check_error.hpp"

namespace ck {

// Initialization flag of Barrier object, can be any value except for zero
static constexpr int BarrierInitFlag = 0x7856;

// 1) only the first thread-block in the synchronizaton group is supposed to call this function. It
// is the responsibility of the user to ensure the two integer values in p_control_bits are zeros
// before calling gms_init().
// 2) Aftercalling gms_reset(), the two integer values in p_control_bits will be zeros, so no
// repetitious initialization of p_control_bits buffer is required
static __device__ void gms_init(int NumWarps, int* p_control_bits)
{
    union
    {
        int two32[2];
        unsigned long one64;
    } regs;

    regs.two32[0] = BarrierInitFlag;
    regs.two32[1] = NumWarps;

    if(threadIdx.x == 0)
        atomicCAS(reinterpret_cast<unsigned long*>(p_control_bits), 0, regs.one64);
};

// all the workgroups in the synchronization group is supposed to call this function
static __device__ void gms_barrier(int* p_control_bits)
{
    constexpr int mask = WarpSize - 1;

    if((threadIdx.x & mask) == 0)
    {
        // ensure the barrier object is initialized
        do
        {
            const int r0 = __atomic_load_n(&p_control_bits[0], __ATOMIC_RELAXED);

            if(r0 == BarrierInitFlag)
                break;

        } while(true);

        // go ahead toward the barrier line
        atomicSub(&p_control_bits[1], 1);

        // wait until all warps have arrived
        do
        {
            const int r1 = __atomic_load_n(&p_control_bits[1], __ATOMIC_RELAXED);

            if(r1 == 0)
                break;

        } while(true);
    };
};

// 1) Only the first thread-block in the synchronizaton group is supposed to call this function.
// 2) Aftercalling gms_reset(), the two integer values in p_control_bits will be zeros, so no
// repetitious initialization of p_control_bits buffer is required
static __device__ void gms_reset(int* p_control_bits)
{
    // reset the barrier object
    if(threadIdx.x == 0)
        (void)atomicCAS(&p_control_bits[0], BarrierInitFlag, 0);
};

} // namespace ck
