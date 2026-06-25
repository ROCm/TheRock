// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"
namespace ck_tile {

template <typename Problem_, typename Policy_>
struct ElementWiseKernel
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType            = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType      = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType            = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using ElementWiseOperation = ck_tile::remove_cvref_t<typename Problem::ElementWiseOperation>;

    static constexpr index_t kBlockSize = Problem::BlockShape::kBlockSize;
    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    template <typename... XDataType, typename Dims>
    CK_TILE_DEVICE void operator()(const Dims lens,
                                   const Dims input_strides,
                                   const Dims output_strides,
                                   const tuple<XDataType...>& input_tensors,
                                   YDataType* p_y) const
    {
        using S = typename Problem::BlockShape;

        // Setup block-level coordinates and transforms
        const index_t iM           = get_block_id() * S::kBlockM;
        const auto merge_transform = make_merge_transform(lens);

        // Load all input tiles into registers.
        // The lambda structure here is intended to minimize the lifetime
        // of intermediate objects (views, windows) used for loading.
        const auto x_tiles = ck_tile::generate_tuple(
            [&](auto i) {
                const auto tensor_view = make_naive_tensor_view<address_space_enum::global>(
                    input_tensors.get(i), lens, input_strides, number<S::kVectorM>{}, number<1>{});

                const auto transformed_tensor = pad_tensor_view(
                    transform_tensor_view(tensor_view,
                                          ck_tile::make_tuple(merge_transform),
                                          ck_tile::make_tuple(make_index_sequence<Dims::size()>{}),
                                          ck_tile::make_tuple(sequence<0>{})),
                    ck_tile::make_tuple(number<S::kBlockM>{}),
                    sequence<Problem::kPad>{});

                const auto x_window =
                    make_tile_window(transformed_tensor,
                                     ck_tile::make_tuple(number<S::kBlockM>{}),
                                     {iM},
                                     Policy::template MakeXBlockTileDistribution<Problem>());

                return load_tile(x_window);
            },
            number<sizeof...(XDataType)>{});

        // Setup output tile in registers.
        const auto& x_tile0 = x_tiles.get(number<0>{});
        auto y_tile = make_static_distributed_tensor<YDataType>(x_tile0.get_tile_distribution());

        // Perform element-wise computation.
        const auto spans = x_tile0.get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx) {
            const auto tile_idx = make_tuple(idx);
            apply(
                [&](auto&&... tiles) {
                    ElementWiseOperation{}(y_tile(tile_idx),
                                           type_convert<ComputeDataType>(tiles[tile_idx])...);
                },
                x_tiles);
        });

        // Setup output window and store the result tile.
        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, lens, output_strides, number<S::kVectorM>{});

        const auto transformed_y_m_n = pad_tensor_view(
            transform_tensor_view(y_m_n,
                                  ck_tile::make_tuple(merge_transform),
                                  ck_tile::make_tuple(make_index_sequence<Dims::size()>{}),
                                  ck_tile::make_tuple(sequence<0>{})),
            ck_tile::make_tuple(number<S::kBlockM>{}),
            sequence<Problem::kPad>{});

        auto y_window = make_tile_window(transformed_y_m_n,
                                         make_tuple(number<S::kBlockM>{}),
                                         {iM},
                                         y_tile.get_tile_distribution());

        store_tile(y_window, cast_tile<YDataType>(y_tile));
    }

    template <typename... Ints>
    CK_TILE_HOST static bool IsSupportedArgument(const ck_tile::tuple<Ints...>& input_sizes)
    {
        // when total elements % kVectorM != 0; should use Pad instead of unsupported
        ignore = input_sizes;
        return true;
    }
};

} // namespace ck_tile
