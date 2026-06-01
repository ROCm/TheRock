// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, float, float>(
        ck::half_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, float, float, float>(
        ck::bhalf_t& e, const float& c, const float& d0, const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, float, float>(
        ck::half_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int, float, float>(
        ck::bhalf_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }
};

template <int KPack, typename T>
void preShuffleBuffer(const T* src, T* dst, int N, int K, int NWmma)
{
    int NLane = NWmma;
    int KLane = ck::get_warp_size() / NLane;

    int K0 = K / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K + k];
        }
    }
}
