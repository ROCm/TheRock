// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

template <auto v>
struct constant
{
    using value_type                  = decltype(v);
    using type                        = constant; // using injected-class-name
    static constexpr value_type value = v;
    CK_TILE_HOST_DEVICE constexpr operator value_type() const noexcept { return value; }
    CK_TILE_HOST_DEVICE constexpr value_type operator()() const noexcept { return value; }
    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return true; }
};

template <auto v>
CK_TILE_HOST_DEVICE static void print(const constant<v>&)
{
    printf("%ld", static_cast<long>(v));
}

template <typename T, T v>
struct integral_constant : constant<v>
{
    using value_type         = T;
    using type               = integral_constant; // using injected-class-name
    static constexpr T value = v;
};

template <index_t v>
using number = constant<v>;

template <long_index_t v>
using long_number = constant<v>;

template <bool b>
using bool_constant = constant<b>;
using true_type     = bool_constant<true>;
using false_type    = bool_constant<false>;

#define CK_TILE_LEFT_UNARY_OP(OP)                               \
    template <auto x>                                           \
    CK_TILE_HOST_DEVICE constexpr auto operator OP(constant<x>) \
    {                                                           \
        return constant<(OP x)>{};                              \
    }

#define CK_TILE_BINARY_OP(OP)                                                \
    template <auto x, auto y>                                                \
    CK_TILE_HOST_DEVICE constexpr auto operator OP(constant<x>, constant<y>) \
    {                                                                        \
        return constant<(x OP y)>{};                                         \
    }

CK_TILE_LEFT_UNARY_OP(+)
CK_TILE_LEFT_UNARY_OP(-)
CK_TILE_LEFT_UNARY_OP(~)
CK_TILE_LEFT_UNARY_OP(!)

CK_TILE_BINARY_OP(+)
CK_TILE_BINARY_OP(-)
CK_TILE_BINARY_OP(*)
CK_TILE_BINARY_OP(/)
CK_TILE_BINARY_OP(%)
CK_TILE_BINARY_OP(&)
CK_TILE_BINARY_OP(|)
CK_TILE_BINARY_OP(^)
CK_TILE_BINARY_OP(<<)
CK_TILE_BINARY_OP(>>)
CK_TILE_BINARY_OP(&&)
CK_TILE_BINARY_OP(||)
CK_TILE_BINARY_OP(==)
CK_TILE_BINARY_OP(!=)
CK_TILE_BINARY_OP(>)
CK_TILE_BINARY_OP(<)
CK_TILE_BINARY_OP(>=)
CK_TILE_BINARY_OP(<=)

#undef CK_TILE_LEFT_UNARY_OP
#undef CK_TILE_BINARY_OP

template <typename T>
struct is_constant : std::false_type
{
};
template <auto v>
struct is_constant<constant<v>> : std::true_type
{
};
template <typename T>
inline constexpr bool is_constant_v = is_constant<T>::value;
} // namespace ck_tile
