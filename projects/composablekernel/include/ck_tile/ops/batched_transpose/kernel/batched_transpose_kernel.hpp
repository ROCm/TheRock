// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

struct BatchedTransposeHostArgs
{
    const void* p_input;
    void* p_output;
    index_t batch;
    index_t height;
    index_t width;
    index_t dim_stride;
    index_t dim_block_h;
    index_t dim_block_w;
};

template <typename Pipeline_>
struct BatchedTransposeKernel
{

    CK_TILE_DEVICE static index_t counter = 0;
    using Pipeline                        = remove_cvref_t<Pipeline_>;
    using Problem                         = remove_cvref_t<typename Pipeline::Problem>;

    using Type = typename Problem::DataType;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    struct BatchedTransposeKargs
    {
        const void* p_input;
        void* p_output;
        index_t batch;
        index_t height;
        index_t width;
        index_t dim_stride;
    };

    using Kargs = BatchedTransposeKargs;
    using Hargs = BatchedTransposeHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& host_args)
    {
        const size_t grid_size_x =
            ck_tile::integer_divide_ceil(host_args.height, host_args.dim_block_h);
        const size_t grid_size_y =
            ck_tile::integer_divide_ceil(host_args.width, host_args.dim_block_w);
        const size_t grid_size_z = host_args.batch;
        return dim3(grid_size_x, grid_size_y, grid_size_z);
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_input    = h.p_input;
        k.p_output   = h.p_output;
        k.batch      = h.batch;
        k.height     = h.height;
        k.width      = h.width;
        k.dim_stride = h.dim_stride;
        return k;
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return Problem::kBlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        static constexpr ck_tile::index_t kMPerBlock         = Problem::kMPerBlock;
        static constexpr ck_tile::index_t kNPerBlock         = Problem::kNPerBlock;
        static constexpr bool kPadM                          = Problem::kPadM;
        static constexpr bool kPadN                          = Problem::kPadN;
        static constexpr ck_tile::index_t VectorSizeInput    = Problem::VectorSizeInput;
        static constexpr ck_tile::index_t VectorStrideInput  = 1;
        static constexpr ck_tile::index_t VectorSizeOutput   = Problem::VectorSizeOutput;
        static constexpr ck_tile::index_t VectorStrideOutput = 1;

        const auto iM     = amd_wave_read_first_lane(blockIdx.x * kMPerBlock);
        const auto iN     = amd_wave_read_first_lane(blockIdx.y * kNPerBlock);
        const auto offset = amd_wave_read_first_lane(blockIdx.z * kargs.height * kargs.width);

        const auto x_m_n = [&]() {
            const auto x_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const Type*>(kargs.p_input) + offset,
                make_tuple(kargs.height, kargs.width),
                make_tuple(kargs.width, 1),
                number<VectorSizeInput>{},
                number<VectorStrideInput>{});

            return pad_tensor_view(x_dram_naive,
                                   make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
                                   sequence<kPadM, kPadN>{});
        }();

        const auto y_n_m = [&]() {
            const auto y_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                static_cast<Type*>(kargs.p_output) + offset,
                make_tuple(kargs.width, kargs.height),
                make_tuple(kargs.height, 1),
                number<VectorSizeOutput>{},
                number<VectorStrideOutput>{});

            return pad_tensor_view(y_dram_naive,
                                   make_tuple(number<kNPerBlock>{}, number<kMPerBlock>{}),
                                   sequence<kPadN, kPadM>{});
        }();

        auto x_block_window = make_tile_window(
            x_m_n,
            make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
            {static_cast<ck_tile::index_t>(iM), static_cast<ck_tile::index_t>(iN)});

        auto y_block_window = make_tile_window(
            y_n_m,
            make_tuple(number<kNPerBlock>{}, number<kMPerBlock>{}),
            {static_cast<ck_tile::index_t>(iN), static_cast<ck_tile::index_t>(iM)});

        Pipeline{}(x_block_window, y_block_window);
    }
};
} // namespace ck_tile
