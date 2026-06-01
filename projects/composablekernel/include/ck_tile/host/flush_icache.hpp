// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

namespace ck_tile {
// GPU kernel to invalidate instruction cache for accurate benchmarking.
// s_icache_inv: Asynchronously invalidates the L1 instruction cache on this compute unit,
//               forcing subsequent kernel runs to fetch instructions from HBM instead of cache.
// 16x s_nop:    Wait cycles (~16 cycles) to ensure cache invalidation completes before kernel
//               exits. Without these NOPs, the flush may not finish, leading to inconsistent
//               timing measurements where some instructions remain cached.
static __global__ void flush_cache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}
} // namespace ck_tile
