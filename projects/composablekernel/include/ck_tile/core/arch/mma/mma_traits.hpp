// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "amdgcn_mma.hpp"
#include "mfma/mfma_traits.hpp"
#include "wmma/wmma_traits.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @class is_mma_op_supported
 * @brief Trait to check if MmaOp is supported
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
// TODO: c++20 template <MmaOpI MmaOp, typename = void>
template <typename MmaOp, typename = void>
struct is_mma_op_supported : std::true_type
{
};

/**
 * @struct is_mma_op_supported
 * @brief The MmaOp is unsupported specialization
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
// TODO: c++20 template <MmaOpI MmaOp>
template <typename MmaOp>
// TODO: c++20 requires
struct is_mma_op_supported<MmaOp,
                           std::enable_if_t<std::is_same_v<typename MmaOp::OpType, Unsupported>>>
    : std::false_type
{
};

/**
 * @brief Convenience evaluation of is_mma_op_supported
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
// TODO: c++20 template <MmaOpI MmaOp>
template <typename MmaOp>
static constexpr bool is_mma_op_supported_v = is_mma_op_supported<MmaOp>::value;

/**
 * @class MmaOpParams
 * @brief Reflects the template parameters of a given MmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
// TODO: c++20 template <MmaOpI MmaOp>
template <typename MmaOp>
struct MmaOpParams;

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L

/**
 *  @concept MmaOpParamsI
 *  @brief  Expresses the required members for each MmaOp
 */
template <typename MmaOpParams>
concept MmaOpParamsI = requires(MmaOpParams op) {
    // Capture template parameters
    typename MmaOpParams::ADataType;
    typename MmaOpParams::BDataType;
    typename MmaOpParams::CDataType;
    typename MmaOpParams::CtrlFlags;

    { MmaOpParams::BlockM } -> std::convertible_to<unsigned int>;
    { MmaOpParams::BlockN } -> std::convertible_to<unsigned int>;
    { MmaOpParams::BlockK } -> std::convertible_to<unsigned int>;
    { MmaOpParams::GfxTargetId } -> std::convertible_to<amdgcn_target_arch_id>;
};

#endif // defined(__cpp_concepts) && __cpp_concepts >= 201907L

/**
 * @struct MmaOpParams
 * @brief Reflects the template parameters of a given MmaOp
 * @tparam ADataType_ Data type of matrix A
 * @tparam BDataType_ Data type of matrix B
 * @tparam CDataType_ Data type of the accumulator
 * @tparam BlockM_ Size of the M dimension
 * @tparam BlockN_ Size of the N dimension
 * @tparam BlockK_ Size of the K dimension
 * @tparam CtrlFlags_ Control flags for the MMA operation
 * @tparam CompilerTarget_ The compiler target
 */
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          uint32_t BlockM_,
          uint32_t BlockN_,
          uint32_t BlockK_,
          typename CtrlFlags_,
          typename CompilerTarget_>
// TODO: c++20 amdgcn_target_arch_id CompilerTarget_>
struct MmaOpParams<amdgcn_mma<ADataType_,
                              BDataType_,
                              CDataType_,
                              BlockM_,
                              BlockN_,
                              BlockK_,
                              CtrlFlags_,
                              CompilerTarget_>>
{
    // Capture incoming template parameters
    using ADataType                  = ADataType_;
    using BDataType                  = BDataType_;
    using CDataType                  = CDataType_;
    static constexpr uint32_t BlockM = BlockM_;
    static constexpr uint32_t BlockN = BlockN_;
    static constexpr uint32_t BlockK = BlockK_;
    using CtrlFlags                  = CtrlFlags_;
    using CompilerTarget             = CompilerTarget_;
    // TODO c++20static constexpr amdgcn_target_arch_id GfxTargetId = CompilerTarget_;
};

/**
 * @class MmaOpTraits
 * @brief Reflects the template parameters and static members of a given MmaOp.
 * @tparam MmaOp The matrix multiply-accumulate operation
 */
template <typename MmaOp>
// TODO: c++20 template <MmaOpI MmaOp>
// TODO: c++20 requires MmaOpParamsI<MmaOpParams<MmaOp>>
struct MmaOpTraits : public MmaOpParams<MmaOp>
{
    // Capture internal MmaOp static members
    using OpType   = typename MmaOp::OpType;
    using AVecType = typename MmaOp::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    // Capture layout parameters
    static constexpr index_t kAMBlock    = MmaOp::kAMBlock;
    static constexpr index_t kBNBlock    = MmaOp::kBNBlock;
    static constexpr index_t kAMLane     = MmaOp::kAMLane;
    static constexpr index_t kBNLane     = MmaOp::kBNLane;
    static constexpr index_t kABKLane    = MmaOp::kABKLane;
    static constexpr index_t kABKPerLane = MmaOp::kABKPerLane;
    static constexpr index_t kCMLane     = MmaOp::kCMLane;
    static constexpr index_t kCNLane     = MmaOp::kCNLane;
    static constexpr index_t kCM0PerLane = MmaOp::kCM0PerLane;
    static constexpr index_t kCM1PerLane = MmaOp::kCM1PerLane;

    // Additional traits to identify the type of MmaOp at compile time
    constexpr static bool IsMfma      = is_mma_op_mfma_v<MmaOp>;
    constexpr static bool IsWmma      = is_mma_op_wmma_v<MmaOp>;
    constexpr static bool IsSupported = is_mma_op_supported_v<MmaOp>;
};

} // namespace ck_tile::core::arch::mma
