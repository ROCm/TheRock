// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/ignore.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct Unsupported
 * @brief  Meta-tag to indicate unsupported amdgcn_mma instance.
 */
struct Unsupported;

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
/**
 * @concept MmaOpI
 * @brief  Expresses the meta-data interface required for each MmaOp policy.
 */
template <typename MmaOp>
concept MmaOpI = requires(MmaOp op) {
    // Requires an op context
    typename MmaOp::OpType;

    // Captures types for inputs / outputs to mma function
    typename MmaOp::AVecType;
    typename MmaOp::BVecType;
    typename MmaOp::CVecType;

    // Captures CK-specific layout properties
    { MmaOp::kAMBlock } -> std::convertible_to<unsigned int>;
    { MmaOp::kBNBlock } -> std::convertible_to<unsigned int>;
    { MmaOp::kAMLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kBNLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kABKLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kABKPerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCMLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCNLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCM0PerLane } -> std::convertible_to<unsigned int>;
    { MmaOp::kCM1PerLane } -> std::convertible_to<unsigned int>;

    // Static exec function
    {
        MmaOp::exec(
            typename MmaOp::AVecType{}, typename MmaOp::BVecType{}, typename MmaOp::CVecType{})
    } -> std::convertible_to<typename MmaOp::CVecType>;
};

#endif // defined(__cpp_concepts) && __cpp_concepts >= 201907L

/**
 *  @class  amdgcn_mma
 *  @brief  This is the default MmaOp policy.
 *          Instances of this class are to be used as MmaOp policies.
 *          Light builtin wrapper for mfma / wmma instructions. This class's job is to
 *          provide a uniform interface to invoke the appropriate instruction
 *          based on the template parameters provided. This interface is to bridge
 *          the gap between the ck_tile API types and the native __builtin types.
 *  @tparam ADataType Datatype of input A
 *  @tparam BDataType Datatype of input B
 *  @tparam CDataType Datatype of accumulator
 *  @tparam BlockM M-dimension of mma block
 *  @tparam BlockN N-dimension of mma block
 *  @tparam BlockK K-dimension of mma block
 *  @tparam CtrlFlags Control flags for mma operation
 *  @tparam CompilerTarget The current compiler target
 *  @tparam Enabler SFINAE enabler
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t BlockM,
          uint32_t BlockN,
          uint32_t BlockK,
          typename CtrlFlags,
          typename CompilerTarget,
          typename Enabler = void>
struct amdgcn_mma
{
    // The base instance is unsupported because there is no __builtin to wrap.
    using OpType = Unsupported;

    // Interface types for A, B, C vectors types
    using AVecType = ext_vector_t<ADataType, 1>;
    using BVecType = ext_vector_t<BDataType, 1>;
    using CVecType = ext_vector_t<CDataType, 1>;

    // Layout constants - default to 0
    static constexpr index_t kAMBlock = 0;
    static constexpr index_t kBNBlock = 0;

    static constexpr index_t kAMLane     = 0;
    static constexpr index_t kBNLane     = 0;
    static constexpr index_t kABKLane    = 0;
    static constexpr index_t kABKPerLane = 0;

    static constexpr index_t kCMLane     = 0;
    static constexpr index_t kCNLane     = 0;
    static constexpr index_t kCM0PerLane = 0;
    static constexpr index_t kCM1PerLane = 0;

    // This is a default pass-through implementation that doesn't do anything practical.
    CK_TILE_DEVICE static CVecType const&
    exec(AVecType const& regsA, BVecType const& regsB, CVecType const& regsC)
    {
        ignore(regsA, regsB);
        return regsC; // No-op, just return C
    }
};

} // namespace ck_tile::core::arch::mma

// Include the implementations
#include "wmma/wmma.hpp"
#include "mfma/mfma.hpp"
