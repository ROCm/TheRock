// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::core::arch::mma {

/**
 * @struct MfmaOp
 * @brief Meta-tag for the MFMA operation. This will be used in the MmaOp policies to
 * identify the operation as an MFMA operation.
 */
struct MfmaOp;

/**
 * @class is_mma_op_mfma
 * @brief Trait to check if MmaOp is an MFMA operation
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp, typename = void>
struct is_mma_op_mfma : std::false_type
{
};

/**
 * @struct is_mma_op_mfma
 * @brief MmaOp specialization for MFMA operations, confirming the OpType matches MfmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
// TODO: c++20 requires
struct is_mma_op_mfma<MmaOp, std::enable_if_t<std::is_same_v<typename MmaOp::OpType, MfmaOp>>>
    : std::true_type
{
};

/**
 * @brief Convenience evaluator for is_mma_op_mfma trait
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
static constexpr bool is_mma_op_mfma_v = is_mma_op_mfma<MmaOp>::value;

} // namespace ck_tile::core::arch::mma
