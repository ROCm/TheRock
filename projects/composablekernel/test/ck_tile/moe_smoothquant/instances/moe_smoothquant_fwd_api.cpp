// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck_tile/core.hpp>
#include "moe_smoothquant.hpp"

template <typename InType,
          typename OutType,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kTwoPass_>
using trait_ = moe_smoothquant_traits_<InType,
                                       OutType,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_N_,
                                       kPadN_,
                                       kTwoPass_>;

template <typename in_type, typename out_type>
float moe_smoothquant_dispatch(moe_smoothquant_args a, const ck_tile::stream_config& s)
{
    float r = -1;
    // clang-format off
    //                                                    rm  rn  tm  tn  vn   pd    2p
    if(a.hidden_size <= 64) {
            r = moe_smoothquant_<trait_<in_type, out_type, 1,  1,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 128) {
        if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type, 1,  1,  4,  64, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type, 1,  2,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 256) {
        if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 1,  4,  64, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2,  4,  64, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 512) {
        if (a.hidden_size % 8 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 1,  4,  64, 8,  true, false>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2,  4,  64, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4,  4,  64, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 8,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 768) {
        if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 3,  4,  64, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 6,  4,  64, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1,12,  4,  64, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 1024) {
        if (a.hidden_size % 8 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 1, 2,  128, 8,  true, false>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2, 2,  128, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4, 2,  128, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4, 1,  256, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 1536) {
        if (a.hidden_size % 8 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 3, 4,   64, 8,  true, false>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 3, 2,  128, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 3, 1,  256, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 6, 1,  256, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 2048) {
        if (a.hidden_size % 8 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 1, 1,  256, 8,  true, false>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2, 1,  256, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4, 1,  256, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 8, 1,  256, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 3072) {
        if (a.hidden_size % 8 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 3, 1,  128, 8,  true, false>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 3, 1,  256, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 6, 1,  256, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 3, 1, 1024, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size <= 4096) {
        if (a.hidden_size % 8 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2, 1,  256, 8,  true, false>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4, 1,  256, 4,  true, false>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2, 1, 1024, 2,  true, false>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4, 1, 1024, 1,  true, false>>(s, a);
    }
    else if(a.hidden_size > 4096) {
        if (a.hidden_size % 8 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2, 1,  256, 8,  true, true>>(s, a);
        else if (a.hidden_size % 4 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4, 1,  256, 4,  true, true>>(s, a);
        else if (a.hidden_size % 2 == 0)
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 2, 1, 1024, 2,  true, true>>(s, a);
        else
            r = moe_smoothquant_<trait_<in_type, out_type,  1, 4, 1, 1024, 1,  true, true>>(s, a);
    }
    return r;
    // clang-format on
}

template <>
float moe_smoothquant<ck_tile::fp16_t, ck_tile::int8_t>(moe_smoothquant_args a,
                                                        const ck_tile::stream_config& s)
{
    return moe_smoothquant_dispatch<ck_tile::fp16_t, ck_tile::int8_t>(a, s);
};

template <>
float moe_smoothquant<ck_tile::fp16_t, ck_tile::fp8_t>(moe_smoothquant_args a,
                                                       const ck_tile::stream_config& s)
{
    return moe_smoothquant_dispatch<ck_tile::fp16_t, ck_tile::fp8_t>(a, s);
};

template <>
float moe_smoothquant<ck_tile::bf16_t, ck_tile::int8_t>(moe_smoothquant_args a,
                                                        const ck_tile::stream_config& s)
{
    return moe_smoothquant_dispatch<ck_tile::bf16_t, ck_tile::int8_t>(a, s);
};

template <>
float moe_smoothquant<ck_tile::bf16_t, ck_tile::fp8_t>(moe_smoothquant_args a,
                                                       const ck_tile::stream_config& s)
{
    return moe_smoothquant_dispatch<ck_tile::bf16_t, ck_tile::fp8_t>(a, s);
};
