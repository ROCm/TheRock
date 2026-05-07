// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename InDataType_,
          typename OutDataType_,
          typename BlockShape_,
          index_t NDimSpatial_,
          index_t AligmentIn_,
          index_t AligmentOut_>
struct BlockImageToColumnProblem
{
    using InDataType  = remove_cvref_t<InDataType_>;
    using OutDataType = remove_cvref_t<OutDataType_>;
    using BlockShape  = remove_cvref_t<BlockShape_>;

    static constexpr index_t NDimSpatial = NDimSpatial_;
    static constexpr index_t AligmentIn  = AligmentIn_;
    static constexpr index_t AligmentOut = AligmentOut_;
};

} // namespace ck_tile
