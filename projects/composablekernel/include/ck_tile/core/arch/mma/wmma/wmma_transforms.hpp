// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/mma_transforms.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct DuplicateTransform
 * @brief Transform to duplicate low register elements to high register elements
 */
struct DuplicateTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType&& v)
    {
        // TODO: Implement duplication logic to broadcast low
        // register elements to high elements [0 - (N/2 -1)] -> [N/2 - (N-1)]
        return std::forward<VecType>(v);
    }
};

/**
 * @struct PadTransform
 * @brief Transform to pad data from original type to b32 type
 */
struct PadTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType&& v)
    {
        // TODO: Implement b32 padding logic.
        // E.g., for fp16, pad each 16-bit element with 16 bits of 0 to make 32-bit elements
        return std::forward<VecType>(v);
    }
};

/**
 * @struct UnpadTransform
 * @brief Transform to unpad data from b32 type to original type
 */
struct UnpadTransform
{
    template <typename VecType>
    CK_TILE_DEVICE static decltype(auto) exec(VecType&& v)
    {
        // TODO: Implement b32 logic to unpad 32 to original data type.
        return std::forward<VecType>(v);
    }
};

/**
 * @struct MmaDefaultTransformsGfx11
 * @brief Default MMA transforms for GFX11 architecture
 */
struct MmaDefaultTransformsGfx11
{
    using ATransform = DuplicateTransform;
    using BTransform = DuplicateTransform;
    using CTransform = PadTransform;
    using DTransform = UnpadTransform;
};

/**
 * @struct MmaDefaultTransformsGfx12
 * @brief Default MMA transforms for GFX12 architecture
 */
struct MmaDefaultTransformsGfx12
{
    using ATransform = PassThroughTransform;
    using BTransform = PassThroughTransform;
    using CTransform = PassThroughTransform;
    using DTransform = PassThroughTransform;
};

/**
 * @struct MmaTransformsDefaultSelector
 * @brief Implements the default MMA transforms selection for gfx11 targets
 * @tparam MmaOp Mma operation
 * @tparam CompilerTarget The compiler target
 */
template <typename MmaOp, typename CompilerTarget>
// TODO: c++20 template <MmaOpI MmaOp, amdgcn_target_arch_id GfxTargetId>
// TODO: c++20 requires
struct MmaTransformsDefaultSelector<MmaOp,
                                    CompilerTarget,
                                    enable_if_target_family_gfx11_t<CompilerTarget>>
{
    using SelectedTransforms = MmaDefaultTransformsGfx11;
};

/**
 * @struct MmaTransformsDefaultSelector
 * @brief Implements the default MMA transforms selection for gfx12 targets
 * @tparam MmaOp Mma operation
 * @tparam CompilerTarget The compiler target
 */
template <typename MmaOp, typename CompilerTarget>
// TODO: c++20 template <MmaOpI MmaOp, amdgcn_target_arch_id GfxTargetId>
// TODO: c++20 requires
struct MmaTransformsDefaultSelector<MmaOp,
                                    CompilerTarget,
                                    enable_if_target_family_gfx12_t<CompilerTarget>>
{
    using SelectedTransforms = MmaDefaultTransformsGfx12;
};

} // namespace ck_tile::core::arch::mma
