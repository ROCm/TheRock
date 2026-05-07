// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <type_traits>
#include <utility>

#include "ck/utility/data_type.hpp"

namespace ck {
namespace utils {

template <typename T>
struct FillUniformDistribution
{
    float a_{-5.f};
    float b_{5.f};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(11939);
        std::uniform_real_distribution<float> dis(a_, b_);
        std::generate(first, last, [&dis, &gen]() { return ck::type_convert<T>(dis(gen)); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

// Normally FillUniformDistributionIntegerValue should use std::uniform_int_distribution as below.
// However this produces segfaults in std::mt19937 which look like inifite loop.
//      template <typename T>
//      struct FillUniformDistributionIntegerValue
//      {
//          int a_{-5};
//          int b_{5};
//
//          template <typename ForwardIter>
//          void operator()(ForwardIter first, ForwardIter last) const
//          {
//              std::mt19937 gen(11939);
//              std::uniform_int_distribution<int> dis(a_, b_);
//              std::generate(
//                  first, last, [&dis, &gen]() { return ck::type_convert<T>(dis(gen)); });
//          }
//      };

// Workaround for uniform_int_distribution not working as expected. See note above.<
template <typename T>
struct FillUniformDistributionIntegerValue
{
    float a_{-5.f};
    float b_{5.f};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(11939);
        std::uniform_real_distribution<float> dis(a_, b_);
        std::generate(
            first, last, [&dis, &gen]() { return ck::type_convert<T>(std::round(dis(gen))); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistributionIntegerValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

/**
 * @brief A functor for filling a container with a monotonically increasing or decreasing sequence.
 *
 * FillMonotonicSeq generates a sequence of values starting from an initial value
 * and incrementing by a fixed step for each subsequent element.
 *
 * @tparam T The numeric type of the sequence elements.
 *
 * Example usage:
 * ```
 * std::vector<int> v(5);
 * FillMonotonicSeq<int>{10, 2}(v); // Fills v with {10, 12, 14, 16, 18}
 * ```
 */
template <typename T>
struct FillMonotonicSeq
{
    T init_value_{0};
    T step_{1};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::generate(first, last, [=, *this, n = init_value_]() mutable {
            auto tmp = n;
            n += step_;
            return tmp;
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillMonotonicSeq&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillConstant
{
    T value_{0};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::fill(first, last, value_);
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillConstant&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct TransformIntoStructuralSparsity
{
    // clang-format off
    static constexpr T valid_sequences[] = {
        0, 0, 1, 1,
        0, 1, 0, 1,
        0, 1, 1, 0,
        1, 0, 0, 1,
        1, 0, 1, 0,
        1, 1, 0, 0,
    };
    // clang-format on

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::for_each(first, last, [=, *this, idx = 0](T& elem) mutable {
            auto tmp_idx = idx;
            idx += 1;
            return elem *= valid_sequences[tmp_idx % (sizeof(valid_sequences) / sizeof(T))];
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const TransformIntoStructuralSparsity&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

} // namespace utils
} // namespace ck
