// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "mfma_traits.hpp"

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"

namespace ck_tile::core::arch::mma {

// NOTE: At this point forward, we are specializing amdgcn_mma for each target id as needed.
// This is because some built-ins are only available on certain target ids.
// We can also do things such add some padding specializations for when we need to use
// smaller values of K that aren't directly supported by the built-ins.
// For flexibility, it is recommended that for each backend wrapper it supports at least
// one packed register for each input to be able to process smaller K values by padding.

/**
 * @struct DefaultMmaCtrlFlags
 * @brief Default MFMA flags, no broadcasting or rotation of inputs
 */
struct DefaultMfmaCtrlFlags
{
    static constexpr uint32_t Cbsz = 0; // CBSZ flag, default 0
    static constexpr uint32_t Abid = 0; // ABID flag, default 0
    static constexpr uint32_t Blgp = 0; // BLGP flag, default 0
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L

/**
 * @concept CtrlFlagsGfx9I
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept CtrlFlagsGfx9I = requires(CtrlFlags ctrlFlags) {
    // Flag members for Gfx9 MFMA instructions
    { CtrlFlags::Cbsz } -> std::convertible_to<int>;
    { CtrlFlags::Abid } -> std::convertible_to<int>;
    { CtrlFlags::Blgp } -> std::convertible_to<int>;
};

#endif // defined(__cpp_concepts) && __cpp_concepts >= 201907L

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for MFMA on GFX9 targets
 *
 * This specialization implements the MFMA instruction for fp16_t A and B
 * matrices, and fp32_t accumulator matrix, with 16x16x16 block sizes.
 *
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
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
                  enable_if_target_family_gfx9_t<CompilerTarget>>
{
    // Mfma operation type
    using OpType = MfmaOp;

    // Register types
    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<fp32_t, 4>;

    // Layout constants
    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x16f16(aVec,
                                                      bVec,
                                                      cVec,
                                                      static_cast<int>(CtrlFlags::Cbsz),
                                                      static_cast<int>(CtrlFlags::Abid),
                                                      static_cast<int>(CtrlFlags::Blgp))};
    }
};

/**
 * @struct amdgcn_mma
 * @brief Specialization of amdgcn_mma for MFMA on GFX950 targets
 *
 * This specialization implements the MFMA instruction for fp16_t A and B
 * matrices, and fp32_t accumulator matrix, with 16x16x32 block sizes.
 *
 * @tparam CtrlFlags Control flags for the MFMA operation
 * @tparam CompilerTarget Current compiler target
 */
// TODO: c++20 template <CtrlFlagsGfx9I CtrlFlags, amdgcn_target CompilerTarget>
// TODO: c++20 requires
template <typename CtrlFlags, typename CompilerTarget>
struct amdgcn_mma<fp16_t,
                  fp16_t,
                  fp32_t,
                  16u,
                  16u,
                  32u,
                  CtrlFlags,
                  CompilerTarget,
                  enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
{
    using OpType = MfmaOp;

    // Packed register types
    using AVecType = ext_vector_t<fp16_t, 8>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<fp32_t, 4>;

    // Layout constants
    static constexpr index_t kAMBlock = 1;
    static constexpr index_t kBNBlock = 1;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 8;
    static constexpr index_t kABKPerLane = 8;

    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    CK_TILE_DEVICE static auto
    exec(AVecType const& aVec, BVecType const& bVec, CVecType const& cVec) -> CVecType
    {
        return {__builtin_amdgcn_mfma_f32_16x16x32_f16(aVec,
                                                       bVec,
                                                       cVec,
                                                       static_cast<int>(CtrlFlags::Cbsz),
                                                       static_cast<int>(CtrlFlags::Abid),
                                                       static_cast<int>(CtrlFlags::Blgp))};
    }
};

} // namespace ck_tile::core::arch::mma
