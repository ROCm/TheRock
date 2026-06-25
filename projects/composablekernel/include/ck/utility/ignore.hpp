// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// https://en.cppreference.com/w/cpp/utility/tuple/ignore

namespace ck {

namespace detail {
struct ignore_t
{
    template <typename T>
    constexpr void operator=(T&&) const noexcept
    {
    }
};
} // namespace detail

inline constexpr detail::ignore_t ignore;

} // namespace ck
