// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ck {
namespace host {

template <class F>
std::string trim(const std::string& s, F f)
{
    auto start = std::find_if_not(s.begin(), s.end(), f);
    auto last  = std::find_if_not(s.rbegin(), std::string::const_reverse_iterator(start), f).base();
    return {start, last};
}

inline std::string trim(const std::string& s)
{
    return trim(s, [](unsigned char c) { return std::isspace(c); });
}

template <class Strings>
inline std::string JoinStrings(Strings strings, const std::string& delim)
{
    auto it = strings.begin();
    if(it == strings.end())
        return "";

    auto nit = std::next(it);
    return std::accumulate(nit, strings.end(), *it, [&](std::string x, std::string y) {
        return std::move(x) + delim + std::move(y);
    });
}

template <class F>
inline std::string
InterpolateString(const std::string& input, F f, std::string start = "${", std::string end = "}")
{
    std::string result = "";
    result.reserve(input.size());
    auto it = input.begin();
    while(it != input.end())
    {
        auto next_start = std::search(it, input.end(), start.begin(), start.end());
        auto next_end   = std::search(next_start, input.end(), end.begin(), end.end());
        result.append(it, next_start);
        if(next_start == input.end())
            break;
        if(next_end == input.end())
        {
            throw std::runtime_error("Unbalanced brackets");
        }
        auto r = f(next_start + start.size(), next_end);
        result.append(r.begin(), r.end());
        it = next_end + end.size();
    }
    return result;
}
inline std::string InterpolateString(const std::string& input,
                                     const std::unordered_map<std::string, std::string>& vars,
                                     std::string start = "${",
                                     std::string end   = "}")
{
    return InterpolateString(
        input,
        [&](auto start_it, auto last_it) {
            auto key = trim({start_it, last_it});
            auto it  = vars.find(key);
            if(it == vars.end())
                throw std::runtime_error("Unknown key: " + key);
            return it->second;
        },
        std::move(start),
        std::move(end));
}

template <class Range, class F>
inline auto Transform(const Range& r, F f) -> std::vector<decltype(f(*r.begin()))>
{
    std::vector<decltype(f(*r.begin()))> result;
    std::transform(r.begin(), r.end(), std::back_inserter(result), f);
    return result;
}

template <class Range1, class Range2, class F>
inline auto Transform(const Range1& r1,
                      const Range2& r2,
                      F f) -> std::vector<decltype(f(*r1.begin(), *r2.begin()))>
{
    std::vector<decltype(f(*r1.begin(), *r2.begin()))> result;
    assert(std::distance(r1.begin(), r1.end()) == std::distance(r2.begin(), r2.end()));
    std::transform(r1.begin(), r1.end(), r2.begin(), std::back_inserter(result), f);
    return result;
}

} // namespace host
} // namespace ck
