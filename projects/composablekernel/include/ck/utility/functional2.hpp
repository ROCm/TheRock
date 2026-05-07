// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/functional.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/type.hpp"

namespace ck {

namespace detail {

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<Sequence<Is...>>
{
    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        swallow{(f(Number<Is>{}), 0)...};
    }
};

} // namespace detail

// F signature: F(Number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    __host__ __device__ constexpr static_for()
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "wrongs! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");
    }

    template <class F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        detail::static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(
            f);
    }
};

namespace detail {

template <typename T, T... Is>
struct applier
{
    template <typename F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        // tweak -fbracket-depth if compilation fails. Clang default limit is 256
        (f(Number<Is>{}), ...);
    }
};

template <int32_t Size> // == sizeof...(Is)
using make_applier = __make_integer_seq<applier, index_t, Size>;

} // namespace detail

template <index_t N>
struct static_for<0, N, 1> : detail::make_applier<N>
{
    using detail::make_applier<N>::operator();
};

template <typename... Is>
struct static_for_range
{
    template <typename F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        // tweak -fbracket-depth if compilation fails. Clang default limit is 256
        (f(Is{}), ...);
    }
};

template <typename... Ts>
struct static_for_product;
template <typename... Is>
struct static_for_product<Tuple<Is...>> : public static_for_range<Is...>
{
};
template <typename... Is, typename... Rest>
struct static_for_product<Tuple<Is...>, Rest...>
{
    template <typename F>
    __host__ __device__ constexpr void operator()(F f) const
    {
        static_for_product<Tuple<Is...>>{}([&](auto i0) {   //
            static_for_product<Rest...>{}([&](auto... is) { //
                f(i0, is...);
            });
        });
    }
};

struct identity
{
    template <typename T>
    __host__ __device__ constexpr T&& operator()(T&& arg) const noexcept
    {
        return ck::forward<T>(arg);
    }
};

} // namespace ck
