// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "ck/utility/functional.hpp"
#include "ck/utility/sequence.hpp"

namespace ck::util {

template <typename Tuple, std::size_t Stride, std::size_t Offset>
struct filter_tuple_by_modulo
{
    // Validate Stride and Offset.
    static_assert(Stride > 0, "Offset must be positive.");
    static_assert(Offset >= 0 && Offset < Stride,
                  "Offset must be positive and less than the stride.");

    // Generate filtered indices for this stride and offset.
    static constexpr int new_size = (std::tuple_size_v<Tuple> + Stride - Offset - 1) / Stride;

    template <std::size_t... Is>
    static constexpr auto to_index(std::index_sequence<Is...>)
    {
        return std::index_sequence<(Offset + Is * Stride)...>{};
    }

    using filtered_indices = decltype(to_index(std::make_index_sequence<new_size>{}));

    // Helper struct to construct the new tuple type from the filtered indices.
    template <typename T, typename Indices>
    struct make_filtered_tuple_type_impl;

    template <typename T, std::size_t... Is>
    struct make_filtered_tuple_type_impl<T, std::index_sequence<Is...>>
    {
        using type = std::tuple<std::tuple_element_t<Is, T>...>;
    };

    using type = typename make_filtered_tuple_type_impl<Tuple, filtered_indices>::type;
};

// Filter a tuple with a stride and offset.
//
// Tuple is a std::tuple or equivalent
// Stride is a positive integer
// Offset is a positive integer smaller than ofset
//
// Evaluates to a smaller tuple type from elements of T with stride M and offset I.
//
// Can be used to filter a tuple of types for sharded instantiations.
template <typename Tuple, std::size_t Stride, std::size_t Offset>
using filter_tuple_by_modulo_t = typename filter_tuple_by_modulo<Tuple, Stride, Offset>::type;

// Example compile-time test:
// using OriginalTuple =
//    std::tuple<int, double, char, float, long, short, bool, char, long long, unsigned int>;
// using NewTuple_Every3rdFrom2nd = filter_tuple_by_modulo_t<OriginalTuple, 3, 1>;
// static_assert(std::is_same_v<NewTuple_Every3rdFrom2nd, std::tuple<double, long, char>>,
//               "Test Case 1 Failed: Every 3rd from 2nd");

} // namespace ck::util
