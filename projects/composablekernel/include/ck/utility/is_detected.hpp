// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/integral_constant.hpp"

namespace ck {

namespace detail {
template <class Default, class AlwaysVoid, template <class...> class Op, class... Args>
struct detector
{
    using value_t = integral_constant<bool, false>;
    using type    = Default;
};

template <class Default, template <class...> class Op, class... Args>
struct detector<Default, ck::void_t<Op<Args...>>, Op, Args...>
{
    using value_t = integral_constant<bool, true>;
    using type    = Op<Args...>;
};
} // namespace detail

struct nonesuch
{
    ~nonesuch()                     = delete;
    nonesuch(nonesuch const&)       = delete;
    void operator=(nonesuch const&) = delete;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <typename T>
using is_pack2_invocable_t = decltype(ck::declval<T&>().is_pack2_invocable);

template <typename T>
using is_pack4_invocable_t = decltype(ck::declval<T&>().is_pack4_invocable);

template <typename T>
using is_pack8_invocable_t = decltype(ck::declval<T&>().is_pack8_invocable);

} // namespace ck
