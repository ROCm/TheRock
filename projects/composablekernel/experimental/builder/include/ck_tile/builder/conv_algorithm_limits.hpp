// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <concepts>

namespace ck_tile::builder {

// Limits for input vector transfer.
template <auto Value>
concept InputVectorTransferLimits = requires {
    requires Value.src_vector_dim > 0 && Value.src_scalar_per_vector > 0 &&
                     Value.lds_dst_scalar_per_vector > 0;
};

// Limits for input and output vector transfer (CK Tile).
template <auto Value>
concept TileInputOutputVectorTransferLimits =
    requires { requires Value.a > 0 && Value.b > 0 && Value.c > 0; };

// Limits for output vector transfer.
template <auto Value>
concept OutputVectorTransferLimits = requires {
    requires Value.scalar_per_vector > 0 && Value.m_xdl_per_wave_per_shuffle > 0 &&
                     Value.n_xdl_per_wave_per_shuffle > 0;
};

// Limits for access order. Must be a permutation of {0, 1, 2}.
template <auto Value>
concept AccessOrderLimits = requires {
    requires((Value[0] != Value[1]) && (Value[0] != Value[2]) && (Value[1] != Value[2]) &&
             (Value[0] >= 0 && Value[0] < 3) && (Value[1] >= 0 && Value[1] < 3) &&
             (Value[2] >= 0 && Value[2] < 3));
};

} // namespace ck_tile::builder
