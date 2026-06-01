// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/topk_softmax/pipeline/topk_softmax_warp_per_row_policy.hpp"
#include <string>
#include <type_traits>

#ifndef TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
#define TOPK_SOFTMAX_USE_RAW_TILE_WINDOW 0
#endif

namespace ck_tile {

template <typename Problem_, typename Policy_ = TopkSoftmaxWarpPerRowPolicy>
struct TopkSoftmaxWarpPerRowPipeline
{
    // TODO: this kernel only support warp per row
    using Problem    = remove_cvref_t<Problem_>;
    using Policy     = remove_cvref_t<Policy_>;
    using WeightType = typename Problem::WeightType;

    template <typename InputWindow, typename OutputWindow, typename IndexWindow>
    CK_TILE_DEVICE auto operator()(const InputWindow& input_window,
                                   OutputWindow& out_window,
                                   IndexWindow& idx_window,
                                   index_t rows,
                                   index_t experts,
                                   index_t k,
                                   index_t block_row_id)
    {
#if TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
        auto inp_win = make_tile_window_linear_raw(
            input_window, Policy::template MakeInputDistribution<Problem>(), sequence<0, 1>{});
#else
        auto inp_win = make_tile_window_linear(
            input_window, Policy::template MakeInputDistribution<Problem>(), sequence<0, 1>{});
#endif
        auto out_win = make_tile_window_linear(out_window.get_bottom_tensor_view(),
                                               out_window.get_window_lengths(),
                                               out_window.get_window_origin(),
                                               Policy::template MakeOutputDistribution<Problem>());
        auto idx_win = make_tile_window_linear(idx_window.get_bottom_tensor_view(),
                                               idx_window.get_window_lengths(),
                                               idx_window.get_window_origin(),
                                               Policy::template MakeOutputDistribution<Problem>());

        auto softmax = Policy::template GetSoftmax<Problem>();
        auto topk    = Policy::template GetTopk<Problem>();

        const index_t grid_rows_per_loop = gridDim.x * Problem::RowsPerBlock;

        while(1)
        {
#if TOPK_SOFTMAX_USE_RAW_TILE_WINDOW
            __builtin_amdgcn_sched_barrier(0);
            auto x =
                load_tile_raw(inp_win, number<-1>{}, bool_constant<true>{}, bool_constant<true>{});
            buffer_load_fence(number<0>{});
            __builtin_amdgcn_sched_barrier(0);
#else
            auto x = load_tile(inp_win);
#endif
            // cast and pad input data
            auto w = [&]() {
#if 0
                auto w_ = cast_tile<WeightType>(x);

                constexpr auto span_2d = decltype(w_)::get_distributed_spans();
                sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        const auto x_indices   = get_x_indices_from_distributed_indices(
                            w_.get_tile_distribution(), i_j_idx);
                        const auto current_expert = x_indices.at(number<1>{});
                        // set to -INF if OOB so that later softmax can work properly
                        w_(i_j_idx) = current_expert >= experts ? -numeric<WeightType>::infinity()
                                                                : w_(i_j_idx);
                    });
                });
                return w_;
#else
                auto w_  = make_static_distributed_tensor<WeightType>(x.get_tile_distribution());
                auto w_f = [&](auto idx) {
                    w_(idx) = type_convert<WeightType>(x(idx));
                    const auto x_indices =
                        get_x_indices_from_distributed_indices(w_.get_tile_distribution(), idx);
                    const auto current_expert = x_indices.at(number<1>{});
                    w_(idx) =
                        current_expert >= experts ? -numeric<WeightType>::infinity() : w_(idx);
                    if constexpr(!Problem::ActivationIsSoftmax)
                    {
                        // sigmoid can be pre-computed already here if not using softmax
                        w_(idx) = WeightType(1) / (WeightType(1) + exp(-w_(idx)));
                    }
                };
                tile_sweeper ts{w_, w_f};
                ts();
                return w_;
#endif
            }();

            if constexpr(Problem::ActivationIsSoftmax)
            {
                auto y = softmax(w);
                topk(y, out_win, idx_win, k);
            }
            else
            {
                // sigmoid was already pre-computed above, so only do topk now
                topk(w, out_win, idx_win, k);
            }

            // check exit
            if constexpr(Problem::LaunchType == 0)
            {
                break;
            }
            else
            {
                block_row_id += grid_rows_per_loop;
                if(block_row_id >= rows)
                    break;
            }

            move_tile_window(inp_win, {grid_rows_per_loop, number<0>{}});
            move_tile_window(out_win, {grid_rows_per_loop, number<0>{}});
            move_tile_window(idx_win, {grid_rows_per_loop, number<0>{}});
        }
    }
};
} // namespace ck_tile
