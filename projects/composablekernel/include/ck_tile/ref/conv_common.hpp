// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include <array>
#include <vector>

namespace ck_tile {

// Helper function to convert std::vector to std::array for kernel parameters
template <ck_tile::index_t NDimSpatial>
inline std::array<ck_tile::long_index_t, NDimSpatial>
to_array(const std::vector<ck_tile::long_index_t>& vec)
{
    std::array<ck_tile::long_index_t, NDimSpatial> arr;
    for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
    {
        arr[i] = vec[i];
    }
    return arr;
}

// Helper to fill missing dimensions with default value
template <ck_tile::index_t NDimSpatial>
inline std::array<ck_tile::long_index_t, NDimSpatial>
to_array_with_default(const std::vector<ck_tile::long_index_t>& vec,
                      ck_tile::long_index_t default_val = 1)
{
    std::array<ck_tile::long_index_t, NDimSpatial> arr;
    for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
    {
        arr[i] = (static_cast<size_t>(i) < vec.size()) ? vec[i] : default_val;
    }
    return arr;
}

// Index calculation helpers for GPU reference kernels
namespace detail {

// Calculate linear input index for grouped convolution
// Layout: [N, spatial..., G, C]
template <index_t NDimSpatial>
inline __device__ long_index_t
calculate_input_index(index_t n,
                      index_t g,
                      index_t c,
                      const std::array<index_t, NDimSpatial>& spatial_idx,
                      const std::array<long_index_t, NDimSpatial + 3>& strides)
{
    long_index_t idx = n * strides[0];
    for(index_t i = 0; i < NDimSpatial; ++i)
        idx += spatial_idx[i] * strides[i + 1];
    idx += g * strides[NDimSpatial + 1] + c;
    return idx;
}

// Calculate linear weight index for grouped convolution
// Layout: [G, K, spatial..., C]
template <index_t NDimSpatial>
inline __device__ long_index_t
calculate_weight_index(index_t g,
                       index_t k,
                       index_t c,
                       const std::array<index_t, NDimSpatial>& spatial_idx,
                       const std::array<long_index_t, NDimSpatial + 3>& strides)
{
    long_index_t idx = g * strides[0] + k * strides[1];
    for(index_t i = 0; i < NDimSpatial; ++i)
        idx += spatial_idx[i] * strides[i + 2];
    idx += c * strides[NDimSpatial + 2];
    return idx;
}

// Calculate linear output index for grouped convolution
// Layout: [N, spatial..., G, K]
template <index_t NDimSpatial>
inline __device__ long_index_t
calculate_output_index(index_t n,
                       index_t g,
                       index_t k,
                       const std::array<index_t, NDimSpatial>& spatial_idx,
                       const std::array<long_index_t, NDimSpatial + 3>& strides)
{
    long_index_t idx = n * strides[0];
    for(index_t i = 0; i < NDimSpatial; ++i)
        idx += spatial_idx[i] * strides[i + 1];
    idx += g * strides[NDimSpatial + 1] + k;
    return idx;
}

} // namespace detail

} // namespace ck_tile
