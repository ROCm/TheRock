// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {

template <auto v>
struct constant
{
    using value_type                  = decltype(v);
    using type                        = constant; // using injected-class-name
    static constexpr value_type value = v;
    __host__ __device__ constexpr operator value_type() const noexcept { return value; }
    __host__ __device__ constexpr value_type operator()() const noexcept { return value; }
};

template <class T, T v>
struct integral_constant : constant<v>
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
};

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator+(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    return integral_constant<decltype(X + Y), X + Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator-(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y <= X, "wrong!");
    return integral_constant<decltype(X - Y), X - Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator*(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    return integral_constant<decltype(X * Y), X * Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator/(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X / Y), X / Y>{};
}

template <typename TX, TX X, typename TY, TY Y>
__host__ __device__ constexpr auto operator%(integral_constant<TX, X>, integral_constant<TY, Y>)
{
    static_assert(Y > 0, "wrong!");
    return integral_constant<decltype(X % Y), X % Y>{};
}

template <bool B>
using bool_constant = integral_constant<bool, B>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;
} // namespace ck
