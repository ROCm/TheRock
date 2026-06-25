// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockAttentionBiasEnum
{
    NO_BIAS          = 0,
    ELEMENTWISE_BIAS = 1, // attention bias, each elements add to the result of Q*K(after scale)
    ALIBI            = 2, // bias computed with position encoding, applied after scale
};

template <BlockAttentionBiasEnum>
struct BlockAttentionBiasEnumToStr;

template <>
struct BlockAttentionBiasEnumToStr<BlockAttentionBiasEnum::NO_BIAS>
{
    static constexpr const char* name = "";
};
template <>
struct BlockAttentionBiasEnumToStr<BlockAttentionBiasEnum::ELEMENTWISE_BIAS>
{
    static constexpr const char* name = "bias";
};
template <>
struct BlockAttentionBiasEnumToStr<BlockAttentionBiasEnum::ALIBI>
{
    static constexpr const char* name = "alibi";
};

} // namespace ck_tile
