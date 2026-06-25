// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

// host side args
struct MoeSmoothquantHostArgs
{
    const void* p_x;        // [tokens ,hidden_size], input, fp16/bf16
    const void* p_smscale;  // [experts, hidden_size], input, columnwise scale, fp32
    const void* p_topk_ids; // [tokens, topk]

    void* p_yscale; // [topk * tokens,  1], output, rowwise quant scale
    void* p_qy;     // [topk * tokens, hidden_size], output

    index_t tokens;
    index_t hidden_size;
    index_t experts;
    index_t topk;
    index_t x_stride; // input x row stride
    index_t y_stride; // output y stride(stride for topk)
};

// TODO: Extract some type to wrapper class
template <typename Pipeline_>
struct MoeSmoothquant
{
    using Pipeline = remove_cvref_t<Pipeline_>;
    using Problem  = typename Pipeline::Problem;

    using XDataType           = remove_cvref_t<typename Problem::XDataType>;
    using SmoothScaleDataType = remove_cvref_t<typename Problem::SmoothScaleDataType>;
    using ComputeDataType     = remove_cvref_t<typename Problem::ComputeDataType>;
    using YScaleDataType      = remove_cvref_t<typename Problem::YScaleDataType>;
    using QYDataType          = remove_cvref_t<typename Problem::QYDataType>;

    static constexpr index_t Block_M = Problem::BlockShape::Block_M;
    static constexpr index_t Block_N = Problem::BlockShape::Block_N;
    static constexpr bool kPadM      = false; // always no need to pad along M
    static constexpr bool kPadN      = Problem::kPadN;
    static constexpr bool kTwoPass   = Problem::kTwoPass;

    static constexpr index_t ThreadPerWarp_N = Problem::BlockShape::ThreadPerWarp_N;
    static constexpr index_t Vector_N        = Problem::BlockShape::Vector_N;
    static constexpr index_t Repeat_N        = Problem::BlockShape::Repeat_N;
    static constexpr index_t kBlockSize      = Problem::BlockShape::BlockSize;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    static_assert(Problem::BlockShape::Repeat_M == 1);

    struct Kargs
    {
        const void* p_x;        // [tokens ,hidden_size], input, fp16/bf16
        const void* p_smscale;  // [experts, hidden_size], input, columnwise scale, fp32
        const void* p_topk_ids; // [tokens, topk]

        void* p_yscale; // [topk, tokens, 1], output, rowwise quant scale
        void* p_qy;     // [topk, tokens, hidden_size], output

        index_t tokens;
        index_t hidden_size;
        index_t experts;
        index_t topk;
        index_t x_stride; // input x row stride
        index_t y_stride; // output y stride(stride for topk)
    };
    using Hargs = MoeSmoothquantHostArgs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& hargs)
    {
        return Kargs{hargs.p_x,
                     hargs.p_smscale,
                     hargs.p_topk_ids,
                     hargs.p_yscale,
                     hargs.p_qy,
                     hargs.tokens,
                     hargs.hidden_size,
                     hargs.experts,
                     hargs.topk,
                     hargs.x_stride,
                     hargs.y_stride};
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        return dim3(hargs.topk, integer_divide_ceil(hargs.tokens, Block_M), 1);
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
    template <> struct t2s<ck_tile::int8_t> { static constexpr const char * name = "i8"; };
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
            if (kTwoPass) n += "_2p";
            return n; }();

        #define _SS_  std::string
        #define _TS_  std::to_string
        return _SS_("moe_smoothquant_") + _SS_(t2s<XDataType>::name) + "_" +  _SS_(t2s<QYDataType>::name) + "_" +
             _TS_(S_::Block_M) + "x" + _TS_(S_::Block_N) + "_" + _TS_(S_::WarpPerBlock_M) + "x" + _TS_(S_::WarpPerBlock_N) + "_" +
             _TS_(S_::Warp_M) + "x" + _TS_(S_::Warp_N) + "_" + _TS_(S_::Vector_M) + "x" + _TS_(S_::Vector_N) + "_" +
             _SS_(Pipeline::name) + surfix;
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t i_topk  = blockIdx.x;
        const index_t i_token = blockIdx.y * Block_M;
        const index_t i_token_in_thrd =
            amd_wave_read_first_lane(threadIdx.x / Problem::BlockShape::ThreadPerBlock_N);

        const index_t i_expert = reinterpret_cast<const index_t*>(
            kargs.p_topk_ids)[(i_token + i_token_in_thrd) * kargs.topk + i_topk];

        // [tokens ,hidden_size]
        const auto x_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const XDataType*>(kargs.p_x),
                make_tuple(kargs.tokens, kargs.hidden_size),
                make_tuple(kargs.x_stride, 1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {i_token, 0});
        }();

        // [experts, hidden_size],
        const auto smscale_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<const SmoothScaleDataType*>(kargs.p_smscale) +
                    i_expert * kargs.hidden_size,
                make_tuple(kargs.hidden_size),
                make_tuple(1),
                number<Vector_N>{},
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_N>{}), sequence<kPadN>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_N>{}), {0});
        }();

        // [topk, tokens]
        auto yscale_window = [&]() {
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<YScaleDataType*>(kargs.p_yscale) + i_topk * kargs.tokens,
                make_tuple(kargs.tokens),
                make_tuple(1),
                number<1>{});

            const auto tmp2_ =
                pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<kPadM>{});

            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {i_token});
        }();

        // [topk, tokens, hidden_size]
        auto qy_window = [&]() {
            auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                static_cast<QYDataType*>(kargs.p_qy) + i_topk * kargs.tokens * kargs.y_stride,
                make_tuple(kargs.tokens, kargs.hidden_size),
                make_tuple(kargs.y_stride, 1),
                number<Vector_N>{},
                number<1>{});

            auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<kPadM, kPadN>{});
            return make_tile_window(
                tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {i_token, 0});
        }();

        __shared__ char smem[GetSmemSize()];

        Pipeline{}(x_window, smscale_window, yscale_window, qy_window, kargs.hidden_size, smem);
    }
};

} // namespace ck_tile
