// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::core::arch::mma {

/**
 * @struct WmmaOp
 * @brief Meta-tag for the WMMA operation. This will be used in the MmaOp struct to
 * identify the operation as an WMMA operation.
 */
struct WmmaOp;

/**
 * @class is_mma_op_wmma
 * @brief Trait to check if MmaOp is an WMMA operation
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp, typename = void>
struct is_mma_op_wmma : std::false_type
{
};

/**
 * @struct is_mma_op_wmma
 * @brief MmaOp specialization for WMMA operations, confirming the OpType matches WmmaOp
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
// TODO: c++20 requires
struct is_mma_op_wmma<MmaOp, std::enable_if_t<std::is_same_v<typename MmaOp::OpType, WmmaOp>>>
    : std::true_type
{
};

/**
 * @brief Convenience evaluator for is_mma_op_wmma trait
 * @tparam MmaOp The matrix multiply-accumulate operation type to check
 */
template <typename MmaOp>
static constexpr bool is_mma_op_wmma_v = is_mma_op_wmma<MmaOp>::value;

} // namespace ck_tile::core::arch::mma
