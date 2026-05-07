// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

/**
 * @brief This T5Pass implements the RMSNorm2d forward pipeline as a variant
 *        based on Rmsnorm2dFwdPipelineOnePass and Rmsnorm2dFwdPipelineTwoPass using a T5 model-like
 * method.
 *
 * The T5 model, developed by Google, is a transformer-based architecture designed to perform
 * a variety of NLP tasks. The T5-like approach employed here is characterized by how RMS
 * normalization is handled, particularly where intermediate values are cast to BF16. This aims to
 * achieve a similar value distribution to that produced by the VLLM hip implementation, thereby
 * enhancing model accuracy.
 *
 * Note: While this implementation improves precision and can reduce discrepancies with VLLM, it is
 * not guaranteed to eliminate all differences or ensure uniform outcomes across every use case.
 *
 * This implementation is a variant based on the original one-pass and two-pass approaches,
 * allowing for both fused and non-fused add operations.
 */

template <typename Problem_, typename Policy_ = Rmsnorm2dFwdPipelineDefaultPolicy>
struct Rmsnorm2dFwdPipelineModelSensitiveT5Pass
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
            [[maybe_unused]] auto pre_out =
                make_static_distributed_tensor<YResidualDataType>(x.get_tile_distribution());

            sweep_tile(x_resi, [&](auto idx) {
                // compute x = x_resi + x
                acc(idx) = type_convert<ComputeDataType>(x_resi(idx)) + acc(idx);

                // To make norm input align with residual output
                if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
                {
                    if constexpr(std::is_same_v<YResidualDataType, ck_tile::bf16_t>)
                    {
                        pre_out(idx) = float_to_bf16<bf16_rounding_mode::standard>(acc(idx));
                    }
                    else
                    {
                        pre_out(idx) = type_convert<YResidualDataType>(acc(idx));
                    }
                    acc(idx) = type_convert<ComputeDataType>(pre_out(idx));
                }
            });
            if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
            {
                store_tile(y_residual_window, pre_out);
            }
        }

        // compute mean square each-thread->cross-lane->cross-warp
        auto square_sum = block_reduce2d.template MakeYBlockTile<decltype(acc)>();
        set_tile(square_sum, 0);
        if constexpr((Problem::BlockShape::Repeat_N * Problem::BlockShape::Vector_N) % 2 == 0)
        {
            sweep_tile(
                acc,
                [&](auto idx_0, auto idx_1) {
                    square_sum(idx_0) += acc[idx_0] * acc[idx_0] + acc[idx_1] * acc[idx_1];
                },
                sequence<1, 2>{});
        }
        else
        {
            square_sum = block_reduce2d(acc,
                                        reduce_square_sum_func.GetIdentityValue<ComputeDataType>(),
                                        reduce_square_sum_func);
        }
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

            if constexpr(std::is_same_v<XDataType, ck_tile::bf16_t>)
            {
                const auto tmp0 =
                    float_to_bf16<bf16_rounding_mode::standard>(acc[idx] * inv_rms_[i_idx]);
                const auto tmp1 = float_to_bf16<bf16_rounding_mode::standard>(
                    type_convert<ComputeDataType>(tmp0) * gamma_);
                const auto rmsn_ = type_convert<ComputeDataType>(tmp1);
                rmsn(idx)        = rmsn_;
            }
            else
            {
                const auto tmp   = type_convert<XDataType>(acc[idx] * inv_rms_[i_idx]);
                const auto rmsn_ = type_convert<ComputeDataType>(tmp) * gamma_;
                rmsn(idx)        = rmsn_;
            }
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
