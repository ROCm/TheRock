// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iterator>
#include <numeric>

namespace ck {
template <typename T, typename ForwardIterator, typename Size, typename BinaryOperation>
auto accumulate_n(ForwardIterator first, Size count, T init, BinaryOperation op)
    -> decltype(std::accumulate(first, std::next(first, count), init, op))
{
    return std::accumulate(first, std::next(first, count), init, op);
}
} // namespace ck
