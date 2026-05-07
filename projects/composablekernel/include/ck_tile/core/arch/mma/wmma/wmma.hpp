// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::core::arch::mma {

/**
 * @enum WmmaCtrlFlags
 * @brief Common wmma control flags for gfx11 and gfx12
 */
enum struct WmmaCtrlFlags : bool
{
    // Only has an effect on gfx11 when the accumulator is 16-bit
    // Determines which half of the 32-bit accum register to use
    // Low = bits [15:0]
    // High = bits[31:16]
    LOW  = false,
    HIGH = true,

    // Only has an effect on gfx11 / 12 when the input is 8-bit int
    // Signage indicator of inputs / accum
    UNSIGNED = false,
    SIGNED   = true
};

} // namespace ck_tile::core::arch::mma

// Include the architecture-specific WMMA implementations and traits
#include "wmma_gfx11.hpp"
#include "wmma_gfx12.hpp"
#include "wmma_selector.hpp"
#include "wmma_traits.hpp"
#include "wmma_transforms.hpp"
