// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

// host side args
// X = A + B, Y = Rmsnorm2d(X), QY = RowwiseDynamicQuant(Y) = SaturateCast(Y / YScale)
struct AddRmsnorm2dRdquantFwdHostArgs
{
    const void* p_a;     // [m ,n], input, fp16/bf16
    const void* p_b;     // [m ,n], input, fp16/bf16
    const void* p_gamma; // [1, n], gamma, prec same as input

    void* p_x;      // [m, n], output, p_a + p_b, fp16/bf16
    void* p_yscale; // [m, 1], output, rowwise quant scale (amax / 127) of reuslt of rmsnorm2d(x)
    void* p_qy;     // [m, n], output, result of quant tensor of rmsnorm2d(x) int8

    float epsilon;

    index_t m;
    index_t n;
    index_t stride; // row_stride
};

// TODO: Extract some type to wrapper class
template <typename Pipeline_>
struct AddRmsnorm2dRdquantFwd
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = typename Pipeline::Problem;

    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using GammaDataType   = remove_cvref_t<typename Problem::GammaDataType>;
    using ComputeDataType = remove_cvref_t<typename Problem::ComputeDataType>;
    using XDataType       = remove_cvref_t<typename Problem::XDataType>;
    using YScaleDataType  = remove_cvref_t<typename Problem::YScaleDataType>;
    using QYDataType      = remove_cvref_t<typename Problem::QYDataType>;

    static constexpr bool kSaveX = Problem::kSaveX;

    static constexpr index_t Block_M = Problem::BlockShape::Block_M;
    static constexpr index_t Block_N = Problem::BlockShape::Block_N;
    static constexpr bool kPadM      = false; // always no need to pad along M
    static constexpr bool kPadN      = Problem::kPadN;
    static constexpr bool kThreePass = Problem::kThreePass;

    static constexpr index_t ThreadPerWarp_N = Problem::BlockShape::ThreadPerWarp_N;
    static constexpr index_t Vector_N        = Problem::BlockShape::Vector_N;
    static constexpr index_t Repeat_N        = Problem::BlockShape::Repeat_N;
    static constexpr index_t kBlockSize      = Problem::BlockShape::BlockSize;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    struct Kargs
    {
        const void* p_a;
        const void* p_b;
        const void* p_gamma;

        void* p_x;
        void* p_yscale;
        void* p_qy;

        float epsilon;

        index_t m;
        index_t n;
        index_t stride; // row_stride
    };
    using Hargs = AddRmsnorm2dRdquantFwdHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return Kargs{hargs.p_a,
                     hargs.p_b,
                     hargs.p_gamma,
                     hargs.p_x,
                     hargs.p_yscale,
                     hargs.p_qy,
                     hargs.epsilon,
                     hargs.m,
                     hargs.n,
                     hargs.stride};
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
    // clang-format on

    // in byte
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Pipeline::GetSmemSize(); }

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        using S_ = typename Problem::BlockShape;
        auto surfix = [&] () {
            std::string n;
            if (kPadN) n += "_pn";
            if (kSaveX) n += "_x";
            if (kThreePass) n += "_2p";
            return n; }();

        #define _SS_  std::string
        #define _TS_  std::to_string
        return _SS_("add_rmsnorm2d_rdquant_fwd_") + _SS_(t2s<XDataType>::name) + "_" +
             _TS_(S_::Block_M) + "x" + _TS_(S_::Block_N) + "_" + _TS_(S_::WarpPerBlock_M) + "x" + _TS_(S_::WarpPerBlock_N) + "_" +
             _TS_(S_::Warp_M) + "x" + _TS_(S_::Warp_N) + "_" + _TS_(S_::Vector_M) + "x" + _TS_(S_::Vector_N) + "_" +
             _SS_(Pipeline::name) + surfix;
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto iM = get_block_id() * Block_M;

        const auto a_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const ADataType*>(kargs.p_a),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        const auto b_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const BDataType*>(kargs.p_b),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
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

        auto x_window = [&]() {
            if constexpr(kSaveX)
            {
                const auto tmp2_ = [&]() {
                    const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                        static_cast<XDataType*>(kargs.p_x),
                        make_tuple(kargs.m, kargs.n),
                        make_tuple(kargs.stride, 1),
                        number<Vector_N>{},
                        number<1>{});

                    return pad_tensor_view(tmp_,
                                           make_tuple(number<Block_M>{}, number<Block_N>{}),
                                           sequence<kPadM, kPadN>{});
                }();
                return make_tile_window(
                    tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
            }
            else
                return make_null_tile_window(make_tuple(number<Block_M>{}, number<Block_N>{}));
        }();

        auto yscale_window = [&]() {
            auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<YScaleDataType*>(kargs.p_yscale),
                make_tuple(kargs.m),
                make_tuple(1),
                number<1>{});

            auto tmp2_ = pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<kPadM>{});
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {iM});
        }();

        auto qy_window = [&]() {
            auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<QYDataType*>(kargs.p_qy),
                make_tuple(kargs.m, kargs.n),
                make_tuple(kargs.stride, 1),
                number<Vector_N>{},
                number<1>{});

            auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {iM, 0});
        }();

        __shared__ char smem[GetSmemSize()];

        Pipeline{}(a_window,
                   b_window,
                   gamma_window,
                   x_window,
                   yscale_window,
                   qy_window,
                   static_cast<const ComputeDataType>(kargs.epsilon),
                   kargs.n,
                   smem);
    }
};

} // namespace ck_tile
