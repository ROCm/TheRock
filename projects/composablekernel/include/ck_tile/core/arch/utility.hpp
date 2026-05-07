// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include <stdint.h>

namespace ck_tile {

// TODO: we have "memory" clobber here because this inline asm is used for async copy
CK_TILE_DEVICE void m0_set_with_memory(index_t v)
{
    asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
}

// NOTE: this is an immediate value
CK_TILE_DEVICE void m0_inc_with_memory(index_t v)
{
    asm volatile("s_add_u32 m0, %0, m0" : : "n"(v) : "memory");
}

template <typename T>
CK_TILE_DEVICE T warp_shuffle_up(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_up(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const uint32_t wrap_around_lane_delta = get_warp_size() - lane_delta;

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (wrap_around_lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}

template <typename T>
CK_TILE_DEVICE T warp_shuffle_down(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_down(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}

template <typename T>
CK_TILE_DEVICE auto warp_shuffle_down_pair(const T& v_local)
{
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const int32x2_t x = __builtin_amdgcn_permlane32_swap(
        bit_cast<int32_t>(v_local), bit_cast<int32_t>(v_local), false, false);

    thread_buffer<T, 2> v;
    v(0) = bit_cast<T>(x[0]);
    v(1) = bit_cast<T>(x[1]);

    return v;
}

template <typename T>
CK_TILE_DEVICE T warp_shuffle(const T& v_local, uint32_t src_lane)
{
#if 0
    return  __shfl(v_local, src_lane);
#elif 1
    if constexpr(sizeof(int32_t) > sizeof(T))
    {
        union packet
        {
            int32_t x;
            T v;
        };
        packet p;
        p.v = v_local;
        packet p_remote;
        p_remote.x = __builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(p));

        return p_remote.v;
    }
    else if constexpr(sizeof(int32_t) == sizeof(T))
    {
        const int32_t v_remote_tmp =
            __builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(v_local));

        return bit_cast<T>(v_remote_tmp);
    }
    else
    {
        static_assert(sizeof(T) % sizeof(int32_t) == 0, "wrong!");
        constexpr index_t elm = sizeof(T) / sizeof(int32_t);
        using vector_type     = thread_buffer<int32_t, elm>;
        auto vs               = bit_cast<vector_type>(v_local);
        auto vs_remote        = vector_type{};
        static_for<0, elm, 1>{}([&](auto i_e) {
            int32_t tmp = __builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(vs[i_e]));
            vs_remote(i_e) = tmp;
        });
        return bit_cast<T>(vs_remote);
    }
#endif
}

template <typename T>
CK_TILE_DEVICE auto flag_to_exec(const T& v_flag)
{
    static_assert(sizeof(T) == 4);
    // per-thread v_flag store into 2x sgpr
    uint32x2_t exec_flag;
    asm volatile("v_cmp_ge_u32 %[s_exec_flag], %[v_flag], 1"
                 : [s_exec_flag] "=s"(exec_flag)
                 : [v_flag] "v"(v_flag));
    return exec_flag;
}

template <typename X, typename Y>
CK_TILE_DEVICE auto cmp_lt_to_exec(const X& x, const Y& y)
{
    static_assert(sizeof(X) == 4 && sizeof(Y) == 4);
    // per-thread cmp store into 2x sgpr
    uint32x2_t exec_flag;
    asm volatile("v_cmp_lt_u32 %[s_exec_flag], %[v_x], %[v_y]"
                 : [s_exec_flag] "=s"(exec_flag)
                 : [v_x] "v"(x), [v_y] "v"(y));
    return exec_flag;
}

} // namespace ck_tile
