// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_traits.hpp"

namespace ck_tile {

// host side args
struct Layernorm2dFwdHostArgs
{
    const void* p_x;          // [m ,n], input, fp16/bf16
    const void* p_x_residual; // [m ,n], shortcut input, prec same as input, nullptr if not used
    const void* p_sm_scale;   // [1 ,n], smooth scale input, fp32, nullptr if not used
    const void* p_x_bias;     // [1, n], bias, prec same as input
    const void* p_gamma;      // [1, n], gamma, prec same as input
    const void* p_beta;       // [1, n], beta, prec same as input

    void* p_y;          // [m, n], output, fp16/bf16
    void* p_y_residual; // [m, n], shortcut output, prec same as input, nullptr if not used
    void* p_y_scale;    // [m, 1], output a dynamic quant per row, nullptr if not used
    void* p_mean;       // [m, 1], output mean, prec same as input, nullptr if not used
    void* p_invStd;     // [m, 1], output inv-stdvariance, prec same as input, nullptr if not used

    float epsilon;

    index_t m;
    index_t n;
    index_t x_stride;  // x row_stride
    index_t xr_stride; // x residule row stride
    index_t y_stride;  // y row stride
    index_t yr_stride; // y residule row stride
};

// TODO: Extract some type to wrapper class
template <typename Pipeline_, typename Epilogue_>
struct Layernorm2dFwd
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Epilogue = remove_cvref_t<Epilogue_>;
    using Problem  = typename Pipeline::Problem;

    using XDataType           = remove_cvref_t<typename Problem::XDataType>;
    using XBiasDataType       = remove_cvref_t<typename Problem::XBiasDataType>;
    using GammaDataType       = remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType        = remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType     = remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType           = remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType        = remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType      = remove_cvref_t<typename Problem::InvStdDataType>;
    using SmoothScaleDataType = remove_cvref_t<typename Problem::SmoothScaleDataType>;
    using YScaleDataType      = remove_cvref_t<typename Problem::YScaleDataType>;

    // for simplicity, shortcut input/output type is same as X
    using XResidualDataType = XDataType;
    using YResidualDataType = XDataType;

    static constexpr bool kHasGamma       = !std::is_same_v<GammaDataType, null_type>;
    static constexpr bool kHasBeta        = !std::is_same_v<BetaDataType, null_type>;
    static constexpr bool kSaveMeanInvStd = Problem::Traits::kSaveMeanInvStd;
    static constexpr bool kSaveMean       = Problem::Traits::kSaveMeanInvStd;
    static constexpr bool kSaveInvStd     = Problem::Traits::kSaveMeanInvStd;

    static constexpr index_t Block_M  = Problem::BlockShape::Block_M;
    static constexpr index_t Block_N  = Problem::BlockShape::Block_N;
    static constexpr bool kPadM       = false; // always no need to pad along M
    static constexpr bool kPadN       = Problem::Traits::kPadN;
    static constexpr bool kTwoPass    = Problem::Traits::kTwoPass;
    static constexpr auto kXbias      = Problem::Traits::kXbias;
    static constexpr auto kFusedAdd   = Problem::Traits::kFusedAdd;
    static constexpr auto kFusedQuant = Problem::Traits::kFusedQuant;

    static constexpr index_t ThreadPerWarp_N = Problem::BlockShape::ThreadPerWarp_N;
    static constexpr index_t Vector_N        = Problem::BlockShape::Vector_N;
    static constexpr index_t Repeat_N        = Problem::BlockShape::Repeat_N;
    static constexpr index_t kBlockSize      = Problem::BlockShape::BlockSize;
    static constexpr auto I0                 = number<0>{};
    static constexpr auto I1                 = number<1>{};

    struct Kargs
    {
        const void* p_x;          // [m ,n], input, fp16/bf16
        const void* p_x_residual; // [m ,n], shortcut input, prec same as input, nullptr if not used
        const void* p_sm_scale;   // [1 ,n], smooth scale input, fp32, nullptr if not used
        const void* p_x_bias;     // [1, n], bias, prec same as input
        const void* p_gamma;      // [1, n], gamma, prec same as input
        const void* p_beta;       // [1, n], beta, prec same as input

        void* p_y;          // [m, n], output, fp16/bf16
        void* p_y_residual; // [m, n], shortcut output, prec same as input, nullptr if not used
        void* p_y_scale;    // [m, 1], output a dynamic quant per row, nullptr if not used

        void* p_mean;   // [m, 1], output mean, prec same as input, nullptr if not used
        void* p_invStd; // [m, 1], output inv-stdvariance, prec same as input, nullptr if not used

        float epsilon;

        index_t m;
        index_t n;
        index_t x_stride;  // x row_stride
        index_t xr_stride; // x residule row stride
        index_t y_stride;  // y row stride
        index_t yr_stride; // y residule row stride
    };
    using Hargs = Layernorm2dFwdHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return Kargs{hargs.p_x,
                     hargs.p_x_residual,
                     hargs.p_sm_scale,
                     hargs.p_x_bias,
                     hargs.p_gamma,
                     hargs.p_beta,
                     hargs.p_y,
                     hargs.p_y_residual,
                     hargs.p_y_scale,
                     hargs.p_mean,
                     hargs.p_invStd,
                     hargs.epsilon,
                     hargs.m,
                     hargs.n,
                     hargs.x_stride,
                     hargs.xr_stride,
                     hargs.y_stride,
                     hargs.yr_stride};
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        return dim3(integer_divide_ceil(hargs.m, Block_M));
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? Problem::BlockShape::template GetBlockSize<true>()
                           : Problem::BlockShape::template GetBlockSize<false>();
    }

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    template <> struct t2s<ck_tile::fp8_t> { static constexpr const char * name = "fp8"; };
    template <> struct t2s<ck_tile::bf8_t> { static constexpr const char * name = "bf8"; };
    template <> struct t2s<ck_tile::int8_t> { static constexpr const char * name = "int8"; };
    // clang-format on

    // in byte
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_HOST static std::string GetName()
    {
#define _SS_ std::string
#define _TS_ std::to_string
        // clang-format off
        using S_ = typename Problem::BlockShape;
        auto surfix = [&] () {
            std::string n;
            if (kXbias != Layernorm2dXBiasEnum::NO_BIAS) n += _SS_("_") + Layernorm2dXBiasEnumName<kXbias>::name;
            if (kFusedAdd != Layernorm2dFusedAddEnum::NO_ADD) n += _SS_("_") + Layernorm2dFusedAddEnumName<kFusedAdd>::name;
            if (kFusedQuant != Layernorm2dFusedQuantEnum::NO_SWEEP) n += _SS_("_") + Layernorm2dFusedQuantEnumName<kFusedQuant>::name;
            if (kPadN) n += "_pn";
            if (kSaveMeanInvStd) n += "_mv";
            // if (kTwoPass) n += "_2p";
            return n; }();

        auto prec_str = [&] () {
            std::string base_str = _SS_(t2s<XDataType>::name);
            if (!std::is_same_v<XDataType, YDataType>) {
                base_str += _SS_("_") + _SS_(t2s<YDataType>::name);
            }
            if (kFusedQuant == Layernorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT) {
                base_str += _SS_("_sx") + _SS_(t2s<SmoothScaleDataType>::name);
                base_str += _SS_("_sy") + _SS_(t2s<YScaleDataType>::name);
            }
            if (kFusedQuant == Layernorm2dFusedQuantEnum::DYNAMIC_QUANT) {
                base_str += _SS_("_sy") + _SS_(t2s<YScaleDataType>::name);
            }
            return base_str;
        }();

        return _SS_("layernorm2d_fwd_") + _SS_(prec_str) + "_" +
             _TS_(S_::Block_M) + "x" + _TS_(S_::Block_N) + "_" + _TS_(S_::WarpPerBlock_M) + "x" + _TS_(S_::WarpPerBlock_N) + "_" +
             _TS_(S_::Warp_M) + "x" + _TS_(S_::Warp_N) + "_" + _TS_(S_::Vector_M) + "x" + _TS_(S_::Vector_N) + "_" +
             _SS_(Pipeline::name) + surfix;
        // clang-format on
#undef _SS_
#undef _TS_
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto iM = get_block_id() * Block_M;

        const auto x_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.x_stride, 1),
                number<Vector_N>{},
                number<1>{});

            // NOTE: we don't do any pad in this kernel for loading, assume that inside kernel will
            // check the max count dynamically
            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<false, false>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        const auto x_residual_window = [&]() {
            if constexpr(kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD_STORE ||
                         kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD)
            {
                const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<const XResidualDataType*>(kargs.p_x_residual),
                    make_tuple(kargs.m, kargs.n),
                    make_tuple(kargs.xr_stride, 1),
                    number<Vector_N>{},
                    number<1>{});

                // NOTE: we don't do any pad in this kernel for loading, assume that inside kernel
                // will check the max count dynamically
                const auto tmp2_ = pad_tensor_view(tmp_,
                                                   make_tuple(number<Block_M>{}, number<Block_N>{}),
                                                   sequence<false, false>{});
                return make_tile_window(
                    tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
            }
            else
            {
                return make_null_tile_window(make_tuple(number<Block_M>{}, number<Block_N>{}));
            }
        }();

        const auto x_bias_window = [&]() {
            if constexpr(kXbias == Layernorm2dXBiasEnum::ADD_BIAS)
            {
                const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<const XBiasDataType*>(kargs.p_x_bias),
                    make_tuple(kargs.n),
                    make_tuple(1),
                    number<Vector_N>{},
                    number<1>{});

                const auto tmp2_ =
                    pad_tensor_view(tmp_, make_tuple(number<Block_N>{}), sequence<false>{});

                return make_tile_window(tmp2_, make_tuple(number<Block_N>{}), {0});
            }
            else
            {
                return make_null_tile_window(make_tuple(number<Block_N>{}));
            }
        }();

        const auto gamma_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const GammaDataType*>(kargs.p_gamma),
                make_tuple(kargs.n),
                make_tuple(1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_N>{}), sequence<false>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_N>{}), {0});
        }();

        const auto beta_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const BetaDataType*>(kargs.p_beta),
                make_tuple(kargs.n),
                make_tuple(1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_N>{}), sequence<false>{});
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {0});
        }();

        auto y_window = [&]() {
            auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<YDataType*>(kargs.p_y),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.y_stride, 1),
                number<Vector_N>{},
                number<1>{});

            auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        auto y_residual_window = [&]() {
            if constexpr(kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD_STORE)
            {
                auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<YResidualDataType*>(kargs.p_y_residual),
                    make_tuple(kargs.m, kargs.n),
                    make_tuple(kargs.yr_stride, 1),
                    number<Vector_N>{},
                    number<1>{});

                auto tmp2_ = pad_tensor_view(tmp_,
                                             make_tuple(number<Block_M>{}, number<Block_N>{}),
                                             sequence<kPadM, kPadN>{});
                return make_tile_window(
                    tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
            }
            else
            {
                return make_null_tile_window(make_tuple(number<Block_M>{}, number<Block_N>{}));
            }
        }();

        auto mean_window = [&]() {
            if constexpr(kSaveMean)
            {
                const auto mean_m = [&]() {
                    const auto mean_dram_naive =
                        make_naive_tensor_view_packed<address_space_enum::global>(
                            static_cast<MeanDataType*>(kargs.p_mean),
                            make_tuple(kargs.m),
                            number<1>{});

                    return pad_tensor_view(
                        mean_dram_naive, make_tuple(number<Block_M>{}), sequence<kPadM>{});
                }();
                return make_tile_window(mean_m, make_tuple(number<Block_M>{}), {iM});
            }
            else
                return make_null_tile_window(make_tuple(number<Block_M>{}));
        }();

        auto inv_std_window = [&]() {
            if constexpr(kSaveInvStd)
            {
                const auto inv_std_m = [&]() {
                    const auto inv_std_dram_naive =
                        make_naive_tensor_view_packed<address_space_enum::global>(
                            static_cast<InvStdDataType*>(kargs.p_invStd),
                            make_tuple(kargs.m),
                            number<1>{});

                    return pad_tensor_view(
                        inv_std_dram_naive, make_tuple(number<Block_M>{}), sequence<kPadM>{});
                }();
                return make_tile_window(inv_std_m, make_tuple(number<Block_M>{}), {iM});
            }
            else
                return make_null_tile_window(make_tuple(number<Block_M>{}));
        }();

        auto sm_scale_window = [&]() {
            if constexpr(kFusedQuant == Layernorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT)
            {
                const auto win_ = [&]() {
                    const auto tmp_0_ = make_naive_tensor_view_packed<address_space_enum::global>(
                        static_cast<const SmoothScaleDataType*>(kargs.p_sm_scale),
                        make_tuple(kargs.n),
                        number<Vector_N>{});

                    return pad_tensor_view(tmp_0_,
                                           make_tuple(number<Block_N>{}),
                                           sequence<false>{}); // sm_scale no need pad
                }();
                return make_tile_window(win_, make_tuple(number<Block_N>{}), {0});
            }
            else
                return make_null_tile_window(make_tuple(number<Block_N>{}));
        }();

        auto y_scale_window = [&]() {
            if constexpr(kFusedQuant == Layernorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT ||
                         kFusedQuant == Layernorm2dFusedQuantEnum::DYNAMIC_QUANT)
            {
                const auto win_ = [&]() {
                    const auto tmp_0_ = make_naive_tensor_view_packed<address_space_enum::global>(
                        static_cast<YScaleDataType*>(kargs.p_y_scale),
                        make_tuple(kargs.m),
                        number<1>{});

                    return pad_tensor_view(
                        tmp_0_, make_tuple(number<Block_M>{}), sequence<kPadM>{});
                }();
                return make_tile_window(win_, make_tuple(number<Block_M>{}), {iM});
            }
            else
                return make_null_tile_window(make_tuple(number<Block_M>{}));
        }();

        __shared__ char smem[GetSmemSize()];

        Pipeline{}(x_window,
                   x_residual_window,
                   x_bias_window,
                   gamma_window,
                   beta_window,
                   y_window,
                   y_residual_window,
                   mean_window,
                   inv_std_window,
                   sm_scale_window,
                   y_scale_window,
                   static_cast<const ComputeDataType>(kargs.epsilon),
                   kargs.n,
                   smem,
                   Epilogue{});
    }
};

} // namespace ck_tile
