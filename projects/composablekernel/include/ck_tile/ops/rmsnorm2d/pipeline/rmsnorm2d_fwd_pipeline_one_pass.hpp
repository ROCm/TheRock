// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Rmsnorm2dFwdPipelineDefaultPolicy>
struct Rmsnorm2dFwdPipelineOnePass
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using InvRmsDataType  = ck_tile::remove_cvref_t<typename Problem::InvRmsDataType>;

    using XResidualDataType = XDataType;
    using YResidualDataType = XDataType;

    static constexpr bool kHasGamma    = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kSaveInvRms  = Problem::Traits::kSaveInvRms;
    static constexpr bool kSaveUnquant = Problem::Traits::kSaveUnquant;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockRmsnorm2dFwdProblem::kPadM
    static constexpr bool kPadN              = Problem::Traits::kPadN;
    static constexpr auto kFusedAdd          = Problem::Traits::kFusedAdd;
    static constexpr auto kFusedQuant        = Problem::Traits::kFusedQuant;

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr_op"; // block per row
        else
            return "wpr_op"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow,
              typename XResidualWindow,
              typename GammaWindow,
              typename YWindow,
              typename YResidualWindow,
              typename InvRmsWindow,
              typename SmoothScaleWindow,
              typename YScaleWindow,
              typename UnquantYWindow,
              typename Epilogue>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const XResidualWindow& x_residual_window_,
                                   const GammaWindow& gamma_window_,
                                   YWindow& y_window_,
                                   const YResidualWindow& y_residual_window_,
                                   InvRmsWindow& inv_rms_window,
                                   const SmoothScaleWindow& sm_scale_window_,
                                   YScaleWindow& y_scale_window_,
                                   UnquantYWindow& unquant_y_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem,
                                   Epilogue) const
    {
        const auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        const auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBlockTileDistribution<Problem>());
        const auto x_residual_window = make_tile_window(
            x_residual_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto y_residual_window = make_tile_window(
            y_residual_window_, Policy::template MakeXBlockTileDistribution<Problem>());

        auto reduce_square_sum_func = ReduceOp::SquareAdd{};
        auto reduce_sum_func        = ReduceOp::Add{};
        auto block_reduce2d         = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync    = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        auto x      = load_tile(x_window);
        auto x_resi = load_tile(x_residual_window);

        // load gamma (TODO: support no gamma?)
        const auto gamma = load_tile(gamma_window);

        auto acc = cast_tile<ComputeDataType>(x);

        if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD ||
                     kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
        {
            sweep_tile(x_resi, [&](auto idx) {
                // compute x = x_resi + x
                acc(idx) = type_convert<ComputeDataType>(x_resi(idx)) + acc(idx);
            });
            if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
            {
                store_tile(y_residual_window, cast_tile<YResidualDataType>(acc));
            }
        }

        // compute mean square each-thread->cross-lane->cross-warp
        auto square_sum = block_reduce2d(acc,
                                         reduce_square_sum_func.GetIdentityValue<ComputeDataType>(),
                                         reduce_square_sum_func);
        block_reduce2d_sync(square_sum, reduce_sum_func);
        block_reduce2d_cross_warp_sync(square_sum, smem, reduce_sum_func);

        // compute inv-rms
        auto inv_rms = tile_elementwise_in(
            [&](const auto& v_) { return rsqrtf(v_ / row_size + epsilon); }, square_sum);

        if constexpr(kSaveInvRms)
            store_tile(inv_rms_window, cast_tile<InvRmsDataType>(inv_rms));

        // rmsnorm computation
        auto rmsn = make_static_distributed_tensor<ComputeDataType>(x.get_tile_distribution());
        sweep_tile(rmsn, [&, inv_rms_ = inv_rms](auto idx) {
            constexpr auto i_idx = make_tuple(idx[number<0>{}]);
            constexpr auto j_idx = make_tuple(idx[number<1>{}]);

            const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);

            auto rmsn_ = acc[idx] * inv_rms_[i_idx] * gamma_;

            rmsn(idx) = rmsn_;
        });

        if constexpr(kFusedQuant == Rmsnorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT)
        {
            if constexpr(kSaveUnquant)
            {
                Epilogue{}(
                    unquant_y_window, y_window_, sm_scale_window_, y_scale_window_, rmsn, smem);
            }
            else
            {
                Epilogue{}(y_window_, sm_scale_window_, y_scale_window_, rmsn, smem);
            }
        }
        else if constexpr(kFusedQuant == Rmsnorm2dFusedQuantEnum::DYNAMIC_QUANT)
        {
            if constexpr(kSaveUnquant)
            {
                Epilogue{}(unquant_y_window, y_window_, y_scale_window_, rmsn, smem);
            }
            else
            {
                Epilogue{}(y_window_, y_scale_window_, rmsn, smem);
            }
        }
        else
        {
            Epilogue{}(y_window_, rmsn, nullptr);
        }
    }
};
} // namespace ck_tile
