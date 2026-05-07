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

struct TopkSoftmaxHostArgs
{
    const void* p_input;
    void* p_output;
    void* p_indices;
    index_t num_rows;
    index_t num_experts;
    index_t topk;
    index_t stride_input;  // row stride for input, at least experts
    index_t stride_output; // row stride for output/indices, at least topk
};

template <typename Pipeline_>
struct TopkSoftmaxKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = remove_cvref_t<typename Pipeline::Problem>;

    using InputType  = typename Problem::InputType;
    using WeightType = typename Problem::WeightType;
    using IndexType  = typename Problem::IndexType;

    static constexpr index_t kBlockSize = Problem::BlockSize;

    struct TopkSoftmaxKargs
    {
        const void* p_input;
        void* p_output;
        void* p_indices;
        index_t num_rows;
        index_t num_experts;
        index_t topk;
        index_t stride_input;  // row stride for input, at least experts
        index_t stride_output; // row stride for output/indices, at least topk
    };

    using Kargs = TopkSoftmaxKargs;
    using Hargs = TopkSoftmaxHostArgs;

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& h)
    {
        if constexpr(Problem::LaunchType > 0)
        {
            int num_cu = [&]() {
                hipDeviceProp_t dev_prop;
                hipDevice_t dev;
                HIP_CHECK_ERROR(hipGetDevice(&dev));
                HIP_CHECK_ERROR(hipGetDeviceProperties(&dev_prop, dev));
                return dev_prop.multiProcessorCount;
            }();
            return dim3(num_cu * Problem::LaunchType);
        }
        else
        {
            const int num_warps = (h.num_rows + Problem::RowsPerWarp - 1) / Problem::RowsPerWarp;
            const int num_blocks =
                (num_warps + Problem::WarpsPerBlock - 1) / Problem::WarpsPerBlock;
            return dim3(num_blocks);
        }
    }

    CK_TILE_HOST static constexpr auto MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.p_input       = h.p_input;
        k.p_output      = h.p_output;
        k.p_indices     = h.p_indices;
        k.num_rows      = h.num_rows;
        k.num_experts   = h.num_experts;
        k.topk          = h.topk;
        k.stride_input  = h.stride_input;
        k.stride_output = h.stride_output;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::BlockSize; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        index_t block_row_id = static_cast<index_t>(blockIdx.x * Problem::RowsPerBlock);

        if(block_row_id > kargs.num_rows)
            return;

        index_t block_os_inp = amd_wave_read_first_lane(block_row_id * kargs.stride_input);
        index_t block_os_out = amd_wave_read_first_lane(block_row_id * kargs.stride_output);
        index_t num_rows_rem = amd_wave_read_first_lane(kargs.num_rows - block_row_id);

        const auto input_window = [&]() {
            const InputType* p_input =
                reinterpret_cast<const InputType*>(kargs.p_input) + block_os_inp;

            auto tmp = make_naive_tensor_view<address_space_enum::global>(
                p_input,
                make_tuple(num_rows_rem, kargs.num_experts),
                make_tuple(kargs.stride_input, 1),
                number<Problem::VectorSize>{},
                number<1>{});

            auto view = pad_tensor_view(
                tmp,
                make_tuple(number<Problem::RowsPerBlock>{}, number<Problem::Experts>{}),
                sequence<0, 1>{}); // out-most dim no need pad(leverage oob)

            return make_tile_window(
                view,
                make_tuple(number<Problem::RowsPerBlock>{}, number<Problem::Experts>{}),
                {0, 0});
        }();

        auto output_window = [&]() {
            WeightType* p_output = reinterpret_cast<WeightType*>(kargs.p_output) + block_os_out;
            auto tmp             = make_naive_tensor_view<address_space_enum::global>(
                p_output,
                make_tuple(num_rows_rem, kargs.topk),
                make_tuple(kargs.stride_output, 1),
                number<Problem::VectorSize>{},
                number<1>{});
            auto view =
                pad_tensor_view(tmp,
                                make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}),
                                sequence<0, 0>{}); // 1. out-most dim no need pad(leverage oob)
                                                   // 2. we loop over topk 1-1, no need padding
            return make_tile_window(
                view, make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}), {0, 0});
        }();

        auto indices_window = [&]() {
            IndexType* p_indices = reinterpret_cast<IndexType*>(kargs.p_indices) + block_os_out;
            auto tmp             = make_naive_tensor_view<address_space_enum::global>(
                p_indices,
                make_tuple(num_rows_rem, kargs.topk),
                make_tuple(kargs.stride_output, 1),
                number<Problem::VectorSize>{},
                number<1>{});
            auto view =
                pad_tensor_view(tmp,
                                make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}),
                                sequence<0, 0>{}); // 1. out-most dim no need pad(leverage oob)
                                                   // 2. we loop over topk 1-1, no need padding
            return make_tile_window(
                view, make_tuple(number<Problem::RowsPerBlock>{}, number<1>{}), {0, 0});
        }();

        Pipeline{}(input_window,
                   output_window,
                   indices_window,
                   kargs.num_rows,
                   kargs.num_experts,
                   kargs.topk,
                   block_row_id);
    }
};
} // namespace ck_tile
