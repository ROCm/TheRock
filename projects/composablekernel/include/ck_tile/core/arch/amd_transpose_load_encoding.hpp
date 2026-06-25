// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"

namespace ck_tile {

// this generate wave level tile distribution
template <typename T, index_t LaneGroupSize = 16, typename = void>
struct LaneGroupTransposeTraits;

template <typename T, index_t LaneGroupSize>
struct LaneGroupTransposeTraits<T, LaneGroupSize, std::enable_if_t<sizeof(T) == 2>>
{
    static_assert(LaneGroupSize == 16 || LaneGroupSize == 32 || LaneGroupSize == 64,
                  "LaneGroupSize must be 16, 32, or 64");
    // before transpose, 4x16
    static constexpr index_t ksecondDim = 4;
    static constexpr index_t kleadDim   = LaneGroupSize;
    // after transpose, 16x4
    static constexpr index_t ksecondDimT = LaneGroupSize;
    static constexpr index_t kleadDimT   = 4;
    template <index_t kOuterDistDim0,
              index_t kOuterDistDim1,
              index_t kInnerDistDim0,
              index_t kInnerDistDim1>
    using TileDistribution = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<kOuterDistDim0, kOuterDistDim1, 4>,
              sequence<kInnerDistDim0, kInnerDistDim1, LaneGroupSize / 16, 4, 4>>,
        tuple<sequence<1, 2, 2, 1, 2>>,
        tuple<sequence<0, 0, 2, 2, 3>>,
        sequence<2, 1, 2>,
        sequence<1, 1, 4>>;
};

template <typename T, index_t LaneGroupSize>
struct LaneGroupTransposeTraits<T, LaneGroupSize, std::enable_if_t<sizeof(T) == 1>>
{
    static constexpr index_t ksecondDim = 8;
    static constexpr index_t kleadDim   = LaneGroupSize;

    static constexpr index_t ksecondDimT = LaneGroupSize;
    static constexpr index_t kleadDimT   = 8;

    template <index_t kOuterDistDim0,
              index_t kOuterDistDim1,
              index_t kInnerDistDim0,
              index_t kInnerDistDim1>
    using TileDistribution = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<kOuterDistDim0, kOuterDistDim1, 8>,
              sequence<kInnerDistDim0, kInnerDistDim1, LaneGroupSize / 16, 2, 8>>,
        tuple<sequence<1, 2, 2, 1, 2>>,
        tuple<sequence<0, 0, 2, 2, 3>>,
        sequence<2, 1, 2>,
        sequence<1, 1, 4>>;
};

/*
 * @brief This function is used to generate the transposed distribution encoding
 *        for the given data type and distribution dimensions.
 *
 * @tparam T The data type of the elements in the tensor.
 * @tparam kOuterDistDim0 The outer distribution dimension 0, which is outer dimension for stride.
 * @tparam kOuterDistDim1 The outer distribution dimension 1, which is inner dimension for stride.
 * @tparam kInnerDistDim0 The inner distribution dimension 0, which is outer dimension for
 * consecutive.
 * @tparam kInnerDistDim1 The inner distribution dimension 1, which is inner dimension for
 * consecutive.
 */
template <typename T,
          index_t LaneGroupSize,
          index_t kOuterDistDim0,
          index_t kOuterDistDim1,
          index_t kInnerDistDim0,
          index_t kInnerDistDim1>
CK_TILE_DEVICE constexpr auto make_transposed_distr_encode()
{
    return typename LaneGroupTransposeTraits<T, LaneGroupSize>::
        template TileDistribution<kOuterDistDim0, kOuterDistDim1, kInnerDistDim0, kInnerDistDim1>{};
}

} // namespace ck_tile
