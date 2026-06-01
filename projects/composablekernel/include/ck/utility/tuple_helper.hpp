// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "functional4.hpp"
#include "tuple.hpp"
#ifndef CK_CODE_GEN_RTC
#include "is_detected.hpp"
#endif

namespace ck {

template <typename F, index_t... ids>
__host__ __device__ constexpr auto generate_tuple_for(F&& f, Sequence<ids...>)
{
    return make_tuple(f(Number<ids>{})...);
}

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tuple(F&& f, Number<N>)
{
    return generate_tuple_for(f, make_index_sequence<N>{});
}

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tuple(F&& f, LongNumber<N>)
{
    return unpack([&f](auto&&... xs) { return make_tuple(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <typename F, index_t N>
__host__ __device__ constexpr auto generate_tie(F&& f, Number<N>)
{
    return unpack([&f](auto&&... xs) { return tie(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

// tx and ty are tuple of references, return type of will tuple of referennce (not rvalue)
template <typename... X, typename... Y>
__host__ __device__ constexpr auto concat_tuple_of_reference(const Tuple<X&...>& tx,
                                                             const Tuple<Y&...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return Tuple<decltype(zs)...>{ck::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

template <typename... X, typename... Y>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2(
        [&](auto... zs) { return Tuple<decltype(zs)...>{ck::forward<decltype(zs)>(zs)...}; },
        tx,
        ty);
}

// Support any number of tuples to concat (also 1)
template <typename... X>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx)
{
    return tx;
}

template <typename... X, typename... Tuples>
__host__ __device__ constexpr auto concat_tuple(const Tuple<X...>& tx, const Tuples&... tuples)
{
    return concat_tuple(tx, concat_tuple(tuples...));
}

namespace detail {

template <typename F, typename X, index_t... Is>
__host__ __device__ constexpr auto transform_tuples_impl(F f, const X& x, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}))...);
}

template <typename F, typename X, typename Y, typename Z, index_t... Is>
__host__ __device__ constexpr auto
transform_tuples_impl(F f, const X& x, const Y& y, const Z& z, Sequence<Is...>)
{
    return make_tuple(f(x.At(Number<Is>{}), y.At(Number<Is>{}), z.At(Number<Is>{}))...);
}

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x)
{
    return detail::transform_tuples_impl(
        f, x, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y)
{
    return detail::transform_tuples_impl(
        f, x, y, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

template <typename F, typename X, typename Y, typename Z>
__host__ __device__ constexpr auto transform_tuples(F f, const X& x, const Y& y, const Z& z)
{
    return detail::transform_tuples_impl(
        f, x, y, z, typename arithmetic_sequence_gen<0, X::Size(), 1>::type{});
}

// By default unroll to the flatten
template <index_t Depth = 0, index_t MaxDepth = -1>
__host__ __device__ constexpr auto UnrollNestedTuple(const Tuple<>& element)
{
    return element;
}

template <index_t Depth = 0, index_t MaxDepth = -1, typename T>
__host__ __device__ constexpr auto UnrollNestedTuple(const T& element)
{
    return make_tuple(element);
}

template <index_t Depth = 0, index_t MaxDepth = -1, typename... Ts>
__host__ __device__ constexpr auto UnrollNestedTuple(const Tuple<Ts...>& tuple)
{
    if constexpr(Depth == MaxDepth)
    {
        return tuple;
    }
    else
    {
        return unpack(
            [&](auto&&... ts) {
                return concat_tuple(UnrollNestedTuple<Depth + 1, MaxDepth>(ts)...);
            },
            tuple);
    }
}

template <typename... Ts>
__host__ __device__ constexpr auto TupleReverse(const Tuple<Ts...>& tuple)
{
    return generate_tuple(
        [&](auto i) {
            using Idx = Number<Tuple<Ts...>::Size() - i - 1>;
            return tuple.At(Idx{});
        },
        Number<Tuple<Ts...>::Size()>{});
}

// Reduce tuple values in specific range using Function
template <index_t Idx, index_t End, typename F, typename... Ts>
__host__ __device__ constexpr auto TupleReduce(F&& f, const Tuple<Ts...>& tuple)
{
    static_assert(Idx < End, "Wrong parameters for TupleReduce");
    if constexpr(Idx + 1 == End)
    {
        return tuple.At(Number<Idx>{});
    }
    else
    {
        return f(tuple.At(Number<Idx>{}), TupleReduce<Idx + 1, End>(f, tuple));
    }
}

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
template <typename T>
using is_tuple = decltype(ck::declval<T&>().IsTuple());
#endif

template <typename... Ts>
__host__ __device__ constexpr auto IsNestedTuple(const Tuple<Ts...>&)
{
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
    return (is_detected<is_tuple, Ts>::value || ...);
#endif
}

template <index_t depth = 0, typename T>
__host__ __device__ constexpr auto TupleDepth(const T&)
{
    return depth;
}

template <index_t depth = 0, typename... Ts>
__host__ __device__ constexpr auto TupleDepth(const Tuple<Ts...>&)
{
    return math::max(TupleDepth<depth + 1>(Ts{})...);
}

template <index_t from, index_t to, typename... Ts>
__host__ __device__ constexpr auto TupleSlice(const Tuple<Ts...>& tuple)
{
    return generate_tuple(
        [&](auto i) {
            using Idx = Number<from + i>;
            return tuple.At(Idx{});
        },
        Number<to - from>{});
}

} // namespace ck
