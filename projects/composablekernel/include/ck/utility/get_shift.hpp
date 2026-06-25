// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {

template <index_t N>
static constexpr __device__ index_t get_shift()
{
    return (get_shift<N / 2>() + 1);
};

template <>
constexpr __device__ index_t get_shift<1>()
{
    return (0);
}

} // namespace ck
