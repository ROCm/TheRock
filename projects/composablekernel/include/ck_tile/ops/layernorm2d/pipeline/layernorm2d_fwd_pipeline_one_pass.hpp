// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_default_policy.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_traits.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Layernorm2dFwdPipelineDefaultPolicy>
struct Layernorm2dFwdPipelineOnePass
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using XBiasDataType   = ck_tile::remove_cvref_t<typename Problem::XBiasDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = ck_tile::remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = ck_tile::remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = ck_tile::remove_cvref_t<typename Problem::InvStdDataType>;

    using XResidualDataType = XDataType;
    using YResidualDataType = XDataType;

    static constexpr bool kHasGamma   = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kHasBeta    = !std::is_same_v<BetaDataType, ck_tile::null_type>;
    static constexpr bool kSaveMean   = Problem::Traits::kSaveMeanInvStd;
    static constexpr bool kSaveInvStd = Problem::Traits::kSaveMeanInvStd;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockLayernorm2dFwdProblem::kPadM
    static constexpr bool kPadN              = Problem::Traits::kPadN;
    static constexpr bool kFastFDiv          = Problem::Traits::kFastFDiv;
    static constexpr bool kWelford           = Problem::Traits::kWelford;
    static constexpr auto kXbias             = Problem::Traits::kXbias;
    static constexpr auto kFusedAdd          = Problem::Traits::kFusedAdd;
    static constexpr auto kFusedQuant        = Problem::Traits::kFusedQuant;

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr"; // block per row
        else
            return "wpr"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow,
              typename XResidualWindow,
              typename XBiasWindow,
              typename GammaWindow,
              typename BetaWindow,
              typename YWindow,
              typename YResidualWindow,
              typename MeanWindow,
              typename InvStdWindow,
              typename SmoothScaleWindow,
              typename YScaleWindow,
              typename Epilogue>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const XResidualWindow& x_residual_window_,
                                   const XBiasWindow& x_bias_window_,
                                   const GammaWindow& gamma_window_,
                                   const BetaWindow& beta_window_,
                                   YWindow& y_window_,
                                   const YResidualWindow& y_residual_window_,
                                   MeanWindow& mean_window,
                                   InvStdWindow& inv_std_window,
                                   const SmoothScaleWindow& sm_scale_window_,
                                   YScaleWindow& y_scale_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem,
                                   Epilogue) const
    {
        const auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        const auto x_bias_window = make_tile_window(
            x_bias_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        const auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        const auto beta_window = make_tile_window(
            beta_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        const auto x_residual_window = make_tile_window(
            x_residual_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto y_residual_window = make_tile_window(
            y_residual_window_, Policy::template MakeXBlockTileDistribution<Problem>());

        auto x            = load_tile(x_window);
        auto x_resi       = load_tile(x_residual_window);
        const auto x_bias = load_tile(x_bias_window);

        int cur_count = 0;
        int max_count =
            block_tile_welford_calculate_max_count<typename Problem::BlockShape>(row_size);
        auto block_norm_reduce      = Policy::template GetBlockNormReduce<Problem>();
        auto block_norm_reduce_sync = Policy::template GetBlockNormReduceSync<Problem>();
        auto block_norm_reduce_cross_warp_sync =
            Policy::template GetBlockNormReduceCrossWarpSync<Problem>();

        using XTensorType = decltype(cast_tile<ComputeDataType>(x));
        auto mean         = block_norm_reduce.template MakeMeanVarBlockTile<XTensorType>();
        auto var          = block_norm_reduce.template MakeMeanVarBlockTile<XTensorType>();
        clear_tile(mean);
        clear_tile(var);
        // load gamma/beta (TODO: support no gamma/beta?)
        const auto gamma = load_tile(gamma_window);
        const auto beta  = load_tile(beta_window);

        auto acc = cast_tile<ComputeDataType>(x);

        if constexpr(kXbias == Layernorm2dXBiasEnum::ADD_BIAS)
        {
            sweep_tile(x, [&](auto idx) {
                // compute x = bias + x
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);
                acc(idx)             = type_convert<ComputeDataType>(x_bias[j_idx]) + acc(idx);
            });
        }

        if constexpr(kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD_STORE ||
                     kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD)
        {
            sweep_tile(x_resi, [&](auto idx) {
                // compute x = x_resi + x
                acc(idx) = type_convert<ComputeDataType>(x_resi(idx)) + acc(idx);
            });
            if constexpr(kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD_STORE)
                store_tile(y_residual_window, cast_tile<YResidualDataType>(acc));
        }

        // compute reduce each-thread->cross-lane->cross-warp
        block_norm_reduce(acc, mean, var, cur_count, max_count);
        block_norm_reduce_sync(mean, var, cur_count);
        block_norm_reduce_cross_warp_sync(mean, var, cur_count, smem);
        if(kWelford)
        {
            block_tile_welford_post_scale_var(var, cur_count, constant<kFastFDiv>{});
        }
        else
        {
            sweep_tile(mean, [&](auto idx) {
                mean(idx) = mean(idx) / type_convert<MeanDataType>(row_size);
                var(idx)  = var(idx) / type_convert<MeanDataType>(row_size) - mean(idx) * mean(idx);
            });
        }
        // compute inv-std
        auto inv_std = tile_elementwise_in(
            [&](const auto& v_) {
                if(kFastFDiv && std::is_same_v<ComputeDataType, float>)
                {
                    return type_convert<ComputeDataType>(1.0f) *
                           __builtin_amdgcn_rcpf(sqrt(v_ + epsilon));
                }
                else
                {
                    return type_convert<ComputeDataType>(1.0f) / sqrt(v_ + epsilon);
                }
            },
            var);

        if constexpr(kSaveMean)
            store_tile(mean_window, cast_tile<MeanDataType>(mean));
        if constexpr(kSaveInvStd)
            store_tile(inv_std_window, cast_tile<InvStdDataType>(inv_std));

        // layernorm computation
        auto ln = make_static_distributed_tensor<ComputeDataType>(acc.get_tile_distribution());
        sweep_tile(ln, [&, mean_ = mean](auto idx) {
            constexpr auto i_idx = make_tuple(idx[number<0>{}]);
            constexpr auto j_idx = make_tuple(idx[number<1>{}]);

            const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);
            const auto beta_  = type_convert<ComputeDataType>(beta[j_idx]);

            auto ln_ = (acc[idx] - mean_[i_idx]) * inv_std[i_idx] * gamma_ + beta_;
            ln(idx)  = ln_;
        });

        if constexpr(kFusedQuant == Layernorm2dFusedQuantEnum::DYNAMIC_QUANT ||
                     kFusedQuant == Layernorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT)
        {
            Epilogue{}(y_window_, sm_scale_window_, y_scale_window, ln, smem);
        }
        else
            Epilogue{}(y_window_, ln, nullptr);
    }
};
} // namespace ck_tile
