// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile::core::arch::mma {

/**
 * @struct MmaDefaultTransformsGfx9
 * @brief Implements the default MMA transforms for gfx9 targets
 */
struct MmaDefaultTransformsGfx9
{
    using ATransform = PassThroughTransform;
    using BTransform = PassThroughTransform;
    using CTransform = PassThroughTransform;
    using DTransform = PassThroughTransform;
};

/**
 * @struct MmaTransformsDefaultSelector
 * @brief Implements the default MMA transforms selection for gfx9 targets
 * @tparam MmaOp Mma operation
 * @tparam CompilerTarget The compiler target
 */
// TODO: c++20 template <MmaOpI MmaOp, amdgcn_target_arch_id CompilerTarget>
// TODO: c++20 requires
template <typename MmaOp, typename CompilerTarget>
struct MmaTransformsDefaultSelector<MmaOp,
                                    CompilerTarget,
                                    enable_if_target_family_gfx9_t<CompilerTarget>>
{
    using SelectedTransforms = MmaDefaultTransformsGfx9;
};

} // namespace ck_tile::core::arch::mma
