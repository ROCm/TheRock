// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "wmma_traits.hpp"

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {
// TODO: Specifically for gfx11 wmma, we need to deal with quirks such as:
//       - Duplicating A and B inputs
//       - Handling C / D is always in b32, even for f16 accumulation.
// NOTE: Two suggestions:
// 1) We could do it here in the wrappers by accepting packed inputs, then swizzling them to
//    duplicate the inputs as needed before calling the actual built-in. This may introduce
//    some instruction overhead and violate single responsibility clauses, but keeps the logic
//    contained within the backend wrapper.
// 2) We could do it at a higher level, e.g. in the Mma interface (workflow) by introducing
//    pre-mma, mma and post-mma steps. The pre-mma step could handle input duplication transform
//    post-mma could implement D-shuffle transform. This may be cleaner and more flexible than
//    trying to handle everything in the backend wrappers.
//
// This current example assumes duplication has already been done, and that C data shuffles have
// already been completed. (e.g. option 2 above). These expect duplicated inputs and pre-shuffled
// data in C.

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things, such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @class DefaultWmmaFlags
 * @brief Generates default WMMA control flags based on data types.
 * @tparam ADataType Data type of matrix A
 * @tparam BDataType Data type of matrix B
 * @tparam CDataType Data type of the accumulator
 */
template <typename ADataType, typename BDataType, typename CDataType>
struct DefaultWmmaCtrlFlags
{
    // Generate default flags for signage
    // Only used currently for integer inputs / accum in gfx11 / gfx12
    constexpr static WmmaCtrlFlags InputSignA =
        std::is_signed_v<ADataType> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;
    constexpr static WmmaCtrlFlags InputSignB =
        std::is_signed_v<BDataType> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;
    constexpr static WmmaCtrlFlags AccumSign =
        std::is_signed_v<CDataType> ? WmmaCtrlFlags::SIGNED : WmmaCtrlFlags::UNSIGNED;

    // Generate default flags for accumulator destination bits.
    // Only used if accumulation size is 16-bit in gfx11
    constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for fp16_t, fp16_t, fp32_t MMA operation on GFX11
 * architecture.
 * @tparam CtrlFlags Control flags for the WMMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx11I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
struct amdgcn_mma<fp16_t,
                  fp16_t,
                  fp32_t,
                  16u,
                  16u,
                  16u,
                  CtrlFlags,
                  CompilerTarget,
                  enable_if_target_family_gfx11_t<CompilerTarget>>
{
    // Wmma operation type
    using OpType = WmmaOp;

    // Register types (duplicated input / b32 accum)
    using AVecType = ext_vector_t<fp16_t, 16>;
    using BVecType = ext_vector_t<fp16_t, 16>;
    using CVecType = ext_vector_t<fp32_t, 8>;

    // Layout constants
    static constexpr index_t kAMBlock    = 1;
    static constexpr index_t kBNBlock    = 1;
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 8;
    static constexpr index_t kABKPerLane = 8;
    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 2;
    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 1;

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aVec, bVec, cVec)};
    }
};

} // namespace ck_tile::core::arch::mma
