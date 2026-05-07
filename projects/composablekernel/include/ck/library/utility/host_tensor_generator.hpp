// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <numeric>
#include <random>

#include "ck/ck.hpp"

template <typename T>
struct GeneratorTensor_0
{
    template <typename... Is>
    T operator()(Is...)
    {
        return T{0};
    }
};

template <typename T>
struct GeneratorTensor_1
{
    T value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return value;
    }
};

template <>
struct GeneratorTensor_1<ck::half_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::half_t operator()(Is...)
    {
        return ck::type_convert<ck::half_t>(value);
    }
};

template <>
struct GeneratorTensor_1<ck::bhalf_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        return ck::type_convert<ck::bhalf_t>(value);
    }
};

#if defined CK_ENABLE_FP8
template <>
struct GeneratorTensor_1<ck::f8_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::f8_t operator()(Is...)
    {
        return ck::type_convert<ck::f8_t>(value);
    }
};

template <>
struct GeneratorTensor_1<ck::bf8_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::bf8_t operator()(Is...)
    {
        return ck::type_convert<ck::bf8_t>(value);
    }
};
#endif

template <>
struct GeneratorTensor_1<ck::f4_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::f4_t operator()(Is...)
    {
        return ck::type_convert<ck::f4_t>(value);
    }
};

template <>
struct GeneratorTensor_1<ck::f4x2_pk_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::f4x2_pk_t operator()(Is...)
    {
        return ck::f4x2_pk_t{ck::type_convert<ck::f4x2_t>(ck::float2_t{value, value})};
    }
};

template <>
struct GeneratorTensor_1<ck::f6x32_pk_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::f6x32_pk_t operator()(Is...)
    {
        ck::f6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            r.pack(ck::type_convert<ck::f6_t>(value), static_cast<ck::index_t>(i));
        });
        return r;
    }
};

template <>
struct GeneratorTensor_1<ck::bf6x32_pk_t>
{
    float value = 1.0;

    template <typename... Is>
    ck::bf6x32_pk_t operator()(Is...)
    {
        ck::bf6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            r.pack(ck::type_convert<ck::bf6_t>(value), static_cast<ck::index_t>(i));
        });
        return r;
    }
};

template <>
struct GeneratorTensor_1<int8_t>
{
    int8_t value = 1;

    template <typename... Is>
    int8_t operator()(Is...)
    {
        return value;
    }
};

template <>
struct GeneratorTensor_1<ck::pk_i4_t>
{
    int8_t value = 1;

    template <typename... Is>
    ck::pk_i4_t operator()(Is...)
    {
        int t         = value + 8;
        ck::pk_i4_t r = ((t << 4) + t) & 0xff;
        return r;
    }
};

template <>
struct GeneratorTensor_1<ck::e8m0_bexp_t>
{
    float value = 1;

    template <typename... Is>
    ck::e8m0_bexp_t operator()(Is...)
    {
        return ck::type_convert<ck::e8m0_bexp_t>(value);
    }
};

template <typename T>
struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        return static_cast<T>((std::rand() % (max_value - min_value)) + min_value);
    }
};

template <>
struct GeneratorTensor_2<ck::f6x32_pk_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::f6x32_pk_t operator()(Is...)
    {
        ck::f6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            float tmp = (std::rand() % (max_value - min_value)) + min_value;
            r.pack(ck::type_convert<ck::f6_t>(tmp), static_cast<ck::index_t>(i));
        });

        return r;
    }
};

template <>
struct GeneratorTensor_2<ck::bf6x32_pk_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::bf6x32_pk_t operator()(Is...)
    {
        ck::bf6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            float tmp = (std::rand() % (max_value - min_value)) + min_value;
            r.pack(ck::type_convert<ck::bf6_t>(tmp), static_cast<ck::index_t>(i));
        });

        return r;
    }
};

template <>
struct GeneratorTensor_2<ck::bhalf_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::bhalf_t>(tmp);
    }
};

template <>
struct GeneratorTensor_2<int8_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    int8_t operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

template <>
struct GeneratorTensor_2<ck::pk_i4_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::pk_i4_t operator()(Is...)
    {
        int hi        = std::rand() % (max_value - min_value) + min_value + 8;
        int lo        = std::rand() % (max_value - min_value) + min_value + 8;
        ck::pk_i4_t r = (((hi & 0xf) << 4) + (lo & 0xf));
        return r;
    }
};

#if defined CK_ENABLE_FP8
template <>
struct GeneratorTensor_2<ck::f8_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::f8_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::f8_t>(tmp);
    }
};
#endif

#if defined CK_ENABLE_BF8
template <>
struct GeneratorTensor_2<ck::bf8_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::bf8_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::bf8_t>(tmp);
    }
};
#endif

template <>
struct GeneratorTensor_2<ck::f4_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::f4_t operator()(Is...)
    {
        float tmp = (std::rand() % (max_value - min_value)) + min_value;
        return ck::type_convert<ck::f4_t>(tmp);
    }
};

template <>
struct GeneratorTensor_2<ck::f4x2_pk_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::f4x2_pk_t operator()(Is...)
    {
        float tmp0 = (std::rand() % (max_value - min_value)) + min_value;
        float tmp1 = (std::rand() % (max_value - min_value)) + min_value;
        return ck::f4x2_pk_t{ck::type_convert<ck::f4x2_t>(ck::float2_t{tmp0, tmp1})};
    }
};

template <typename T>
struct GeneratorTensor_3
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        return static_cast<T>(min_value + tmp * (max_value - min_value));
    }
};

template <>
struct GeneratorTensor_3<ck::bhalf_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::bhalf_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::bhalf_t>(fp32_tmp);
    }
};

#if defined CK_ENABLE_FP8
template <>
struct GeneratorTensor_3<ck::f8_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::f8_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::f8_t>(fp32_tmp);
    }
};
#endif

#if defined CK_ENABLE_BF8
template <>
struct GeneratorTensor_3<ck::bf8_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::bf8_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::bf8_t>(fp32_tmp);
    }
};
#endif

template <>
struct GeneratorTensor_3<ck::f4_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::f4_t operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp = min_value + tmp * (max_value - min_value);

        return ck::type_convert<ck::f4_t>(fp32_tmp);
    }
};

template <>
struct GeneratorTensor_3<ck::f4x2_pk_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::f4x2_pk_t operator()(Is...)
    {
        float tmp0 = float(std::rand()) / float(RAND_MAX);
        float tmp1 = float(std::rand()) / float(RAND_MAX);

        float fp32_tmp0 = min_value + tmp0 * (max_value - min_value);
        float fp32_tmp1 = min_value + tmp1 * (max_value - min_value);

        return ck::f4x2_pk_t{ck::type_convert<ck::f4x2_t>(ck::float2_t{fp32_tmp0, fp32_tmp1})};
    }
};

template <>
struct GeneratorTensor_3<ck::pk_i4_t>
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    ck::pk_i4_t operator()(Is...)
    {
        int hi        = std::rand() % (max_value - min_value) + min_value + 8;
        int lo        = std::rand() % (max_value - min_value) + min_value + 8;
        ck::pk_i4_t r = (((hi & 0xf) << 4) + (lo & 0xf));
        return r;
    }
};

template <>
struct GeneratorTensor_3<ck::f6x32_pk_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::f6x32_pk_t operator()(Is...)
    {
        ck::f6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            float rnd  = float(std::rand()) / float(RAND_MAX);
            float fp32 = min_value + rnd * (max_value - min_value);
            r.pack(ck::type_convert<ck::f6_t>(fp32), static_cast<ck::index_t>(i));
        });

        return r;
    }
};

template <>
struct GeneratorTensor_3<ck::bf6x32_pk_t>
{
    float min_value = 0;
    float max_value = 1;

    template <typename... Is>
    ck::bf6x32_pk_t operator()(Is...)
    {
        ck::bf6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            float rnd  = float(std::rand()) / float(RAND_MAX);
            float fp32 = min_value + rnd * (max_value - min_value);
            r.pack(ck::type_convert<ck::bf6_t>(fp32), static_cast<ck::index_t>(i));
        });

        return r;
    }
};

template <typename T>
struct GeneratorTensor_4
{
    std::mt19937 generator;
    std::normal_distribution<float> distribution;

    GeneratorTensor_4(float mean, float stddev, unsigned int seed = 1)
        : generator(seed), distribution(mean, stddev){};

    template <typename... Is>
    T operator()(Is...)
    {
        float tmp = distribution(generator);

        return ck::type_convert<T>(tmp);
    }
};

template <>
struct GeneratorTensor_4<ck::f4x2_pk_t>
{
    std::mt19937 generator;
    std::normal_distribution<float> distribution;

    GeneratorTensor_4(float mean, float stddev, unsigned int seed = 1)
        : generator(seed), distribution(mean, stddev){};

    template <typename... Is>
    ck::f4x2_pk_t operator()(Is...)
    {
        float fp32_tmp0 = distribution(generator);
        float fp32_tmp1 = distribution(generator);

        return ck::f4x2_pk_t{ck::type_convert<ck::f4x2_t>(ck::float2_t{fp32_tmp0, fp32_tmp1})};
    }
};

template <>
struct GeneratorTensor_4<ck::f6x32_pk_t>
{
    std::mt19937 generator;
    std::normal_distribution<float> distribution;

    GeneratorTensor_4(float mean, float stddev, unsigned int seed = 1)
        : generator(seed), distribution(mean, stddev){};

    template <typename... Is>
    ck::f6x32_pk_t operator()(Is...)
    {
        ck::f6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            r.pack(ck::type_convert<ck::f6_t>(distribution(generator)),
                   static_cast<ck::index_t>(i));
        });

        return r;
    }
};

template <>
struct GeneratorTensor_4<ck::bf6x32_pk_t>
{
    std::mt19937 generator;
    std::normal_distribution<float> distribution;

    GeneratorTensor_4(float mean, float stddev, unsigned int seed = 1)
        : generator(seed), distribution(mean, stddev){};

    template <typename... Is>
    ck::bf6x32_pk_t operator()(Is...)
    {
        ck::bf6x32_pk_t r;
        ck::static_for<0, 32, 1>{}([&](auto i) {
            r.pack(ck::type_convert<ck::bf6_t>(distribution(generator)),
                   static_cast<ck::index_t>(i));
        });

        return r;
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {static_cast<ck::index_t>(Xs)...};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, ck::index_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

/**
 * @brief Is used to generate sequential values based on the specified dimension.
 *
 * @tparam T The type of the tensor values.
 * @tparam Dim The specific dimension used for generation.
 *
 * GeneratorTensor_Sequential<1>{} will generate the following values for a 3x3 tensor:
 *
 * 0 1 2
 * 0 1 2
 * 0 1 2
 *
 * Essentially, the values generated are logical coordinates of the generated element that
 * correspond to dimension Dim. E.g. for 2-dimensional tensor and Dim=1, the values are the column
 * indices.
 *
 */
template <typename T, ck::index_t Dim>
struct GeneratorTensor_Sequential
{
    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        float tmp = dims[Dim];
        return ck::type_convert<T>(tmp);
    }
};

template <ck::index_t Dim>
struct GeneratorTensor_Sequential<ck::f4x2_pk_t, Dim>
{
    template <typename... Ts>
    ck::f4x2_pk_t operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        float tmp = dims[Dim];
        return ck::type_convert<ck::f4x2_t>(ck::float2_t(tmp));
    }
};

template <ck::index_t Dim>
struct GeneratorTensor_Sequential<ck::f6x32_pk_t, Dim>
{
    template <typename... Ts>
    ck::f6x32_pk_t operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        float tmp = dims[Dim];

        ck::f6x32_pk_t r;
        ck::static_for<0, 32, 1>{}(
            [&](auto i) { r.pack(ck::type_convert<ck::f6_t>(tmp), static_cast<ck::index_t>(i)); });
        return r;
    }
};

template <ck::index_t Dim>
struct GeneratorTensor_Sequential<ck::bf6x32_pk_t, Dim>
{
    template <typename... Ts>
    ck::bf6x32_pk_t operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};

        float tmp = dims[Dim];

        ck::bf6x32_pk_t r;
        ck::static_for<0, 32, 1>{}(
            [&](auto i) { r.pack(ck::type_convert<ck::bf6_t>(tmp), static_cast<ck::index_t>(i)); });
        return r;
    }
};

template <typename T, size_t NumEffectiveDim = 2>
struct GeneratorTensor_Diagonal
{
    T value{1};

    template <typename... Ts>
    T operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        size_t start_dim                            = dims.size() - NumEffectiveDim;
        bool pred                                   = true;
        for(size_t i = start_dim + 1; i < dims.size(); i++)
        {
            pred &= (dims[start_dim] == dims[i]);
        }
        return pred ? value : T{0};
    }
};
