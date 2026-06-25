// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
namespace element_wise {

struct Add
{
    template <typename Y, typename X0, typename X1>
    __host__ __device__ constexpr void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const half_t& x1) const
    {
        y = x0 + type_convert<half_t>(x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const float& x1) const
    {
        y = type_convert<half_t>(x0 + x1);
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const float& x0, const half_t& x1) const
    {
        y = type_convert<half_t>(x0) + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const bf16_t& x1) const
    {
        const float x1_tmp = type_convert<float>(x1);
        y                  = x0 + x1_tmp;
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bf16_t>(bf16_t& y, const bf16_t& x0, const bf16_t& x1) const
    {
        const float x1_tmp = type_convert<float>(x0);
        const float x2_tmp = type_convert<float>(x1);
        const float y_tmp  = x1_tmp + x2_tmp;
        y                  = type_convert<bf16_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bf16_t>(bf16_t& y, const float& x0, const bf16_t& x1) const
    {
        const float x2_tmp = type_convert<float>(x1);
        const float y_tmp  = x0 + x2_tmp;
        y                  = type_convert<bf16_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<bf16_t>(bf16_t& y, const float& x0, const float& x1) const
    {
        const float y_tmp = x0 + x1;
        y                 = type_convert<bf16_t>(y_tmp);
    }

    template <>
    __host__ __device__ constexpr void
    operator()<int8_t>(int8_t& y, const int8_t& x0, const int8_t& x1) const
    {
        y = x0 + x1;
    };
};

} // namespace element_wise
} // namespace ck_tile
