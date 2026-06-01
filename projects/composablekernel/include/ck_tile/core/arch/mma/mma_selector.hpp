// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
namespace ck_tile::core::arch::mma {

/**
 * @class MmaDefaultSelector
 * @brief Implements a default mma selector strategy for the current target architecture.
 * This is simply intended as a default selection strategy for mma instruction operations.
 * Given the particular datatypes and Fragment dimensions, the selector will attempt to
 * select the instruction with the largest K dimension that is supported on the current target
 * architecture.
 * @tparam ADataType       Data type of matrix A
 * @tparam BDataType       Data type of matrix B
 * @tparam CDataType     Data type of the accumulator
 * @tparam FragM           Fragment M dimension
 * @tparam FragN           Fragment N dimension
 * @tparam FragK           Fragment K dimension
 * @tparam CompilerTarget  The compiler target
 * @tparam Enable          SFINAE enabler
 * @note Here we distinguish that Fragment MNK sizes from Block MNK sizes used in the actual MMA
 * operation. Fragment sizes correspond to the overall tile size being computed, while Block sizes
 * correspond to the size of the individual MMA instructions being used to compute the overall in
 * block-wise. The Fragment sizes must be multiples of the Block sizes and in general larger than or
 * equal to the Block sizes.
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t FragM,
          uint32_t FragN,
          uint32_t FragK,
          typename CompilerTarget,
          typename Enable = void>
// TODO c++20 requires
struct MmaDefaultSelector
{
    // By default, no selection is made, and we fall back to a pass-through unsupported
    // implementation. This is because we do not have any knowledge of the target architecture here.
    using SelectedOp =
        amdgcn_mma<ADataType, BDataType, CDataType, FragM, FragN, FragK, void, amdgcn_target<>>;
};

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L

/**
 *  @concept MmaSelectorI
 *  @brief  Expresses the required members for each MmaSelector class.
 */
template <typename MmaSelector>
concept MmaSelectorI = requires(MmaSelector op) {
    // Selectors should have a resulting SelectedOp type
    typename MmaSelector::SelectedOp;
};

#endif // defined(__cpp_concepts) && __cpp_concepts >= 201907L

} // namespace ck_tile::core::arch::mma

// Include the implementations
#include "wmma/wmma_selector.hpp"
#include "mfma/mfma_selector.hpp"
