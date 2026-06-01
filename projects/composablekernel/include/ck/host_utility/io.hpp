// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include <iterator>

#include "ck/tensor_description/tensor_descriptor.hpp"

namespace ck {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(os, " "));
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& v)
{
    std::copy(std::begin(v), std::end(v), std::ostream_iterator<T>(os, " "));
    return os;
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, const TensorDescriptor<Ts...>& desc)
{
    constexpr index_t nDim = remove_cvref_t<decltype(desc)>::GetNumOfDimension();

    os << "{";

    static_for<0, nDim - 1, 1>{}([&](auto i) { os << desc.GetLength(i) << ", "; });

    os << desc.GetLength(Number<nDim - 1>{});

    os << "}";

    return os;
}

} // namespace ck
