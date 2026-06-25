// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_traits.hpp"

namespace ck_tile {

// host side args
struct Rmsnorm2dFwdHostArgs
{
    const void* p_x;          // [m ,n], input, fp16/bf16
    const void* p_x_residual; // [m ,n], shortcut input, prec same as input, nullptr if not used
    const void* p_sm_scale;   // [1 ,n], smooth scale input, fp32, nullptr if not used
    const void* p_gamma;      // [1, n], gamma, prec same as input

    void* p_y;          // [m, n], output, fp16/bf16
    void* p_y_residual; // [m, n], shortcut output, prec same as input, nullptr if not used
    void* p_y_scale;    // [m, 1], output a dynamic quant per row, nullptr if not used
    void* p_invRms;     // [m, 1], output inv-rms, prec same as input, nullptr if not used
    void* p_y_unquant;  // [m, n], output result before quant, nullptr if not used

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
struct Rmsnorm2dFwd
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Epilogue = remove_cvref_t<Epilogue_>;
    using Problem  = typename Pipeline::Problem;

    using XDataType           = remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType       = remove_cvref_t<typename Problem::GammaDataType>;
    using ComputeDataType     = remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType           = remove_cvref_t<typename Problem::YDataType>;
    using InvRmsDataType      = remove_cvref_t<typename Problem::InvRmsDataType>;
    using SmoothScaleDataType = remove_cvref_t<typename Problem::SmoothScaleDataType>;
    using YScaleDataType      = remove_cvref_t<typename Problem::YScaleDataType>;
    using UnquantYDataType    = remove_cvref_t<typename Problem::UnquantYDataType>;

    // for simplicity, shortcut input/output type is same as X
    using XResidualDataType = XDataType;
    using YResidualDataType = XDataType;

    static constexpr bool kHasGamma    = !std::is_same_v<GammaDataType, null_type>;
    static constexpr bool kSaveInvRms  = Problem::Traits::kSaveInvRms;
    static constexpr bool kSaveUnquant = Problem::Traits::kSaveUnquant;

    static constexpr index_t Block_M                = Problem::BlockShape::Block_M;
    static constexpr index_t Block_N                = Problem::BlockShape::Block_N;
    static constexpr bool kPadM                     = false; // always no need to pad along M
    static constexpr bool kPadN                     = Problem::Traits::kPadN;
    static constexpr bool kTwoPass                  = Problem::Traits::kTwoPass;
    static constexpr auto kFusedAdd                 = Problem::Traits::kFusedAdd;
    static constexpr auto kFusedQuant               = Problem::Traits::kFusedQuant;
    static constexpr auto kUseModelSensitiveRMSNorm = Problem::Traits::kUseModelSensitiveRMSNorm;

    static constexpr index_t ThreadPerWarp_N = Problem::BlockShape::ThreadPerWarp_N;
    static constexpr index_t Vector_N        = Problem::BlockShape::Vector_N;
    static constexpr index_t Repeat_N        = Problem::BlockShape::Repeat_N;
    static constexpr index_t kBlockSize      = Problem::BlockShape::BlockSize;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    struct Kargs
    {
        const void* p_x;
        const void* p_x_residual;
        const void* p_sm_scale;
        const void* p_gamma;

        void* p_y;
        void* p_y_residual;
        void* p_y_scale;
        void* p_invRms;
        void* p_y_unquant;

        float epsilon;

        index_t m;
        index_t n;
        index_t x_stride;  // x row_stride
        index_t xr_stride; // x residule row stride
        index_t y_stride;  // y row stride
        index_t yr_stride; // y residule row stride
    };
    using Hargs = Rmsnorm2dFwdHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return Kargs{hargs.p_x,
                     hargs.p_x_residual,
                     hargs.p_sm_scale,
                     hargs.p_gamma,
                     hargs.p_y,
                     hargs.p_y_residual,
                     hargs.p_y_scale,
                     hargs.p_invRms,
                     hargs.p_y_unquant,
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
            if (kFusedAdd != Rmsnorm2dFusedAddEnum::NO_ADD) n += _SS_("_") + Rmsnorm2dFusedAddEnumName<kFusedAdd>::name;
            if (kFusedQuant != Rmsnorm2dFusedQuantEnum::NO_SWEEP) n += _SS_("_") + Rmsnorm2dFusedQuantEnumName<kFusedQuant>::name;
            if (kPadN) n += "_pn";
            if (kSaveInvRms) n += "_rms";
            if (kTwoPass) n += "_2p";
            if (kUseModelSensitiveRMSNorm == Rmsnorm2dSensitiveEnum::NO_SPECIFIC_MODEL) n += "_nsm";
            else if (kUseModelSensitiveRMSNorm == Rmsnorm2dSensitiveEnum::T5_MODEL_LIKE) n += "_t5ml";
            return n; }();

        auto prec_str = [&] () {
            std::string base_str = _SS_(t2s<XDataType>::name);
            if (!std::is_same_v<XDataType, YDataType>) {
                base_str += _SS_("_") + _SS_(t2s<YDataType>::name);
            }
            if (kFusedQuant == Rmsnorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT) {
                base_str += _SS_("_sx") + _SS_(t2s<SmoothScaleDataType>::name);
                base_str += _SS_("_sy") + _SS_(t2s<YScaleDataType>::name);
            }
            if (kFusedQuant == Rmsnorm2dFusedQuantEnum::DYNAMIC_QUANT) {
                base_str += _SS_("_sy") + _SS_(t2s<YScaleDataType>::name);
            }
            return base_str;
        }();

        return _SS_("rmsnorm2d_fwd_") + _SS_(prec_str) + "_" +
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

            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        const auto x_residual_window = [&]() {
            if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD ||
                         kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
            {
                const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<const XResidualDataType*>(kargs.p_x_residual),
                    make_tuple(kargs.m, kargs.n),
                    make_tuple(kargs.xr_stride, 1),
                    number<Vector_N>{},
                    number<1>{});

                const auto tmp2_ = pad_tensor_view(tmp_,
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

        const auto gamma_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const GammaDataType*>(kargs.p_gamma),
                make_tuple(kargs.n),
                make_tuple(1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_N>{}), sequence<kPadN>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_N>{}), {0});
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
            if constexpr(kFusedAdd == Rmsnorm2dFusedAddEnum::PRE_ADD_STORE)
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

        auto inv_rms_window = [&]() {
            if constexpr(kSaveInvRms)
            {
                const auto inv_rms_m = [&]() {
                    const auto inv_rms_dram_naive =
                        make_naive_tensor_view_packed<address_space_enum::global>(
                            static_cast<InvRmsDataType*>(kargs.p_invRms),
                            make_tuple(kargs.m),
                            number<1>{});

                    return pad_tensor_view(
                        inv_rms_dram_naive, make_tuple(number<Block_M>{}), sequence<kPadM>{});
                }();
                return make_tile_window(inv_rms_m, make_tuple(number<Block_M>{}), {iM});
            }
            else
                return make_null_tile_window(make_tuple(number<Block_M>{}));
        }();

        auto sm_scale_window = [&]() {
            if constexpr(kFusedQuant == Rmsnorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT)
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
            {
                return make_null_tile_window(make_tuple(number<Block_N>{}));
            }
        }();

        auto y_scale_window = [&]() {
            if constexpr(kFusedQuant == Rmsnorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT ||
                         kFusedQuant == Rmsnorm2dFusedQuantEnum::DYNAMIC_QUANT)
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
            {
                return make_null_tile_window(make_tuple(number<Block_M>{}));
            }
        }();

        auto unquant_y_window = [&]() {
            if constexpr((kFusedQuant == Rmsnorm2dFusedQuantEnum::SMOOTH_DYNAMIC_QUANT ||
                          kFusedQuant == Rmsnorm2dFusedQuantEnum::DYNAMIC_QUANT) &&
                         kSaveUnquant)
            {
                auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                    static_cast<UnquantYDataType*>(kargs.p_y_unquant),
                    make_tuple(kargs.m, kargs.n),
                    make_tuple(kargs.y_stride, 1),
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

        __shared__ char smem[GetSmemSize()];

        Pipeline{}(x_window,
                   x_residual_window,
                   gamma_window,
                   y_window,
                   y_residual_window,
                   inv_rms_window,
                   sm_scale_window,
                   y_scale_window,
                   unquant_y_window,
                   static_cast<const ComputeDataType>(kargs.epsilon),
                   kargs.n,
                   smem,
                   Epilogue{});
    }
};

} // namespace ck_tile
