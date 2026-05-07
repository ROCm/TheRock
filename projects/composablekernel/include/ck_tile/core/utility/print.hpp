// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

namespace str_literal_detail {
template <size_t... Idx>
constexpr std::tuple<std::integral_constant<size_t, Idx>...>
makeTuple(std::index_sequence<Idx...>) noexcept
{
    return {};
}
constexpr size_t constexpr_strlen(const char* c)
{
    size_t t = 0;
    while(*c++)
        ++t;
    return t;
}
} // namespace str_literal_detail

template <char... Xs>
struct str_literal
{
    static constexpr const char data[] = {Xs..., '\0'};
    static constexpr const size_t size = sizeof...(Xs);

    template <char... Ys>
    CK_TILE_HOST_DEVICE constexpr auto operator+(str_literal<Ys...> /*rhs*/) const
    {
        return str_literal<Xs..., Ys...>{};
    }

    template <size_t N, char... Ys>
    CK_TILE_HOST_DEVICE static constexpr auto duplicate_n(const str_literal<Ys...> sep)
    {
        if constexpr(N == 0)
            return str_literal<>{};
        else if constexpr(N == 1)
            return str_literal<Xs...>{};
        else
            return duplicate_n<N - 1>(sep) + str_literal<Ys..., Xs...>{};
    }
};

#define make_str_literal(lit_)                                                                     \
    std::apply([](auto... indices) { return str_literal<(lit_)[decltype(indices)::value]...>{}; }, \
               str_literal_detail::makeTuple(                                                      \
                   std::make_index_sequence<str_literal_detail::constexpr_strlen(lit_)>()))

/// Declare a ck_tile::print() interface that gets specialized in each header file for types that
/// can be printed.
template <typename T>
CK_TILE_HOST_DEVICE void print(const T&)
{
    static_assert(sizeof(T) == 0,
                  "No print implementation available for this type. Please specialize "
                  "ck_tile::print for your type.");
}

/// Specialization for int
template <>
CK_TILE_HOST_DEVICE void print(const int& value)
{
    printf("%d", value);
}

/// Specialization for float
template <>
CK_TILE_HOST_DEVICE void print(const float& value)
{
    printf("%f", value);
}

/// Specialization for double
template <>
CK_TILE_HOST_DEVICE void print(const double& value)
{
    printf("%f", value);
}

/// Specialization for long
template <>
CK_TILE_HOST_DEVICE void print(const long& value)
{
    printf("%ld", value);
}

/// Specialization for unsigned int
template <>
CK_TILE_HOST_DEVICE void print(const unsigned int& value)
{
    printf("%u", value);
}

/// Specialization for char
template <>
CK_TILE_HOST_DEVICE void print(const char& value)
{
    printf("%c", value);
}

/// Specialization for array
template <typename T, size_t N>
CK_TILE_HOST_DEVICE void print(const T (&value)[N])
{
    printf("[");
    for(size_t i = 0; i < N; ++i)
    {
        if(i > 0)
            printf(", ");
        print(value[i]); // Recursively call print for each element
    }
    printf("]");
}

} // namespace ck_tile
