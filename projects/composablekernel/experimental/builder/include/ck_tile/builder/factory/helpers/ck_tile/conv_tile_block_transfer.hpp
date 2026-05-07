// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory::internal {

struct TileScalarPerVector
{
    size_t a = 0;
    size_t b = 0;
    size_t c = 0;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr TileScalarPerVector SetTileBlockTransfer()
{
    return TileScalarPerVector{.a = ALGORITHM.transfer.a_scalar_per_vector,
                               .b = ALGORITHM.transfer.b_scalar_per_vector,
                               .c = ALGORITHM.transfer.c_scalar_per_vector};
}

} // namespace ck_tile::builder::factory::internal
