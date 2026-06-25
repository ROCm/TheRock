// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {

#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#ifdef __gfx12__
__device__ void llvm_amdgcn_s_wait_dscnt(short cnt) __asm("llvm.amdgcn.s.wait.dscnt");
#endif
#endif

__device__ void block_sync_lds()
{
#if CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#if defined(__gfx12__)
    llvm_amdgcn_s_wait_dscnt(0);
    asm volatile("s_barrier_signal -1\n\t"
                 "s_barrier_wait -1");
#elif defined(__gfx11__)
    // asm volatile("\
    // s_waitcnt lgkmcnt(0) \n \
    // s_barrier \
    // " ::);
    __builtin_amdgcn_s_waitcnt(0xfc07);
    __builtin_amdgcn_s_barrier();
#else
    // asm volatile("\
    // s_waitcnt lgkmcnt(0) \n \
    // s_barrier \
    // " ::);
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
#endif
#else
    __syncthreads();
#endif
}

__device__ void block_sync_lds_direct_load()
{
#ifdef __gfx12__
    asm volatile("\
    s_wait_loadcnt 0x0 \n \
    s_wait_dscnt 0x0 \n \
    s_barrier_signal -1 \n \
    s_barrier_wait -1 \
    " ::);
#else
    asm volatile("\
    s_waitcnt vmcnt(0) \n \
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
#endif
}

__device__ void s_nop()
{
#if 1
    asm volatile("\
    s_nop 0 \n \
    " ::);
#else
    __builtin_amdgcn_sched_barrier(0);
#endif
}

} // namespace ck
