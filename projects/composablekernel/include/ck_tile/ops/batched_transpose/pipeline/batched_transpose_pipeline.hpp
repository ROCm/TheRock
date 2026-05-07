// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BatchedTransposePolicy>
struct BatchedTransposePipeline
{
    // TODO: this kernel only support warp per row
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    template <typename InputWindow, typename OutputWindow>
    CK_TILE_DEVICE auto operator()(const InputWindow& input_window, OutputWindow& out_window)
    {
        auto inp_win =
            make_tile_window(input_window, Policy::template MakeInputDistribution<Problem>());

        auto input_tile = load_tile(inp_win);

        auto output_tile = make_static_distributed_tensor<typename Problem::DataType>(
            Policy::template MakeOutputDistribution<Problem>());

        transpose_tile2d(output_tile, input_tile);

        auto out_win =
            make_tile_window(out_window, Policy::template MakeOutputDistribution<Problem>());

        store_tile(out_win, output_tile);
    }
};
} // namespace ck_tile
