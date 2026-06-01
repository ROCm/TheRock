// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType, typename BDataType>
void reference_transpose_elementwise(const HostTensor<ADataType>& a, HostTensor<BDataType>& b)
{
    ck_tile::index_t M = static_cast<ck_tile::index_t>(a.mDesc.get_lengths()[0]);
    ck_tile::index_t N = static_cast<ck_tile::index_t>(a.mDesc.get_lengths()[1]);

    // Ensure the b tensor is sized correctly for N x M
    if(static_cast<ck_tile::index_t>(b.mDesc.get_lengths()[0]) != N ||
       static_cast<ck_tile::index_t>(b.mDesc.get_lengths()[1]) != M)
    {
        throw std::runtime_error("Output tensor b has incorrect dimensions for transpose.");
    }

    auto f = [&](auto i, auto j) {
        auto v_a = a(i, j);
        b(j, i)  = ck_tile::type_convert<BDataType>(v_a);
    };

    make_ParallelTensorFunctor(f, M, N)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
