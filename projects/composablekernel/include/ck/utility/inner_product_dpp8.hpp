// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "amd_gemm_dpp.hpp"
#include "data_type.hpp"
#include "type_convert.hpp"

namespace ck {

namespace dpp8 {

/// Number of lanes that can share data using DPP8 modifiers.
constexpr index_t lane_group_size = 8;

template <int SrcLaneIdx>
__device__ void inline_v_dot2c_dpp8_instr(const half2_t& a, const half2_t& b, float& c);

// clang-format off
template <>
__device__ void inline_v_dot2c_dpp8_instr<0>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[0, 0, 0, 0, 0, 0, 0, 0]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<1>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[1, 1, 1, 1, 1, 1, 1, 1]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<2>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[2, 2, 2, 2, 2, 2, 2, 2]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<3>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[3, 3, 3, 3, 3, 3, 3, 3]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<4>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[4, 4, 4, 4, 4, 4, 4, 4]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<5>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[5, 5, 5, 5, 5, 5, 5, 5]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<6>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[6, 6, 6, 6, 6, 6, 6, 6]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
template <>
__device__ void inline_v_dot2c_dpp8_instr<7>(const half2_t& a, const half2_t& b, float& c){
    asm volatile("\n v_dot2c_f32_f16_dpp %0, %1, %2 dpp8:[7, 7, 7, 7, 7, 7, 7, 7]" : "=v"(c) : "v"(a), "v"(b), "0"(c));
}
// clang-format on

/**
 * Dot product of two vectors using `v_dot` instruction with DPP8 submitted as inline assembly.
 */
template <int SrcLaneIdx, bool ShareA>
__device__ void inline_v_dot2c_dpp8(const half2_t& a, const half2_t& b, float& c)
{
    static_assert(SrcLaneIdx >= 0 && SrcLaneIdx < dpp8::lane_group_size,
                  "DPP8 src broadcast lane out of range <0, 7>.");
    if constexpr(ShareA)
    {
        inline_v_dot2c_dpp8_instr<SrcLaneIdx>(a, b, c);
    }
    else
    {
        inline_v_dot2c_dpp8_instr<SrcLaneIdx>(b, a, c);
    }
}

/**
 * DPP8 instrinsics expects to get an integer mask, hardcoding integers for specific broadcast
 * patters.
 */
constexpr std::array<int, dpp8::lane_group_size> IntrinsicMaskDpp8 = {
    0,        // 0, 0, 0, 0, 0, 0, 0, 0
    2396745,  // 1, 1, 1, 1, 1, 1, 1, 1
    4793490,  // 2, 2, 2, 2, 2, 2, 2, 2
    7190235,  // 3, 3, 3, 3, 3, 3, 3, 3
    9586980,  // 4, 4, 4, 4, 4, 4, 4, 4
    11983725, // 5, 5, 5, 5, 5, 5, 5, 5
    14380470, // 6, 6, 6, 6, 6, 6, 6, 6
    16777215, // 7, 7, 7, 7, 7, 7, 7, 7
};

/**
 * Returns DPP8 sel modifier as an integer required for the intrinsic instruction.
 */
template <int SrcLaneIdx>
constexpr int get_dpp_sel_mask_broadcast()
{
    static_assert(SrcLaneIdx >= 0 && SrcLaneIdx < dpp8::lane_group_size,
                  "DPP8 src broadcast lane out of range <0, 7>.");
    return IntrinsicMaskDpp8[SrcLaneIdx];
}

template <int SrcLaneIdx>
__device__ void intrinsic_fdot2_impl(const half2_t& a, const half2_t& b, float& c)
{
    constexpr int sel_mask = get_dpp_sel_mask_broadcast<SrcLaneIdx>();
    const half2_t val_from_other_lane =
        bit_cast<half2_t>(__builtin_amdgcn_mov_dpp8(bit_cast<int>(a), sel_mask));
    c = __builtin_amdgcn_fdot2(val_from_other_lane, b, c, false);
}

/**
 * Dot product of two vectors using `v_dot` instruction with DPP8 submitted using intrinsics.
 */
template <int SrcLaneIdx, bool ShareA>
__device__ void intrinsic_fdot2(const half2_t& a, const half2_t& b, float& c)
{
    if constexpr(ShareA)
    {
        intrinsic_fdot2_impl<SrcLaneIdx>(a, b, c);
    }
    else
    {
        intrinsic_fdot2_impl<SrcLaneIdx>(b, a, c);
    }
}

/**
 * Dot product of two input vectors `a`, `b` using `v_dot` instructions with DPP modifier.
 *
 * DPP modifier allows us to share one of the vectors between lanes in a lane group.
 * When `ShareA` is set, instruction uses vector `a` from lane `SrcLaneIdx` from the same
 * lane group (8 lanes per lane group in DPP8). When `ShareA` is not set, vector `b` is shared.
 * Note that all the threads in a lane group uses the same vector - broadcast pattern.
 *
 * `SrcLaneIdx` must be in range from 0 to 7.
 */
template <typename TA, typename TB, typename TC, int SrcLaneIdx, bool ShareA>
__device__ void inner_product_dpp(const TA& a, const TB& b, TC& c)
{
#if CK_USE_AMD_V_DOT_DPP8_INLINE_ASM
    inline_v_dot2c_dpp8<SrcLaneIdx, ShareA>(a, b, c);
#else
    intrinsic_fdot2<SrcLaneIdx, ShareA>(a, b, c);
#endif
}

} // namespace dpp8

} // namespace ck
