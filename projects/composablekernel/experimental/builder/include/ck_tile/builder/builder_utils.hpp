// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/sequence.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {

// Convert a static array to a sequence
// Usage example:
// static constexpr std::vector arr {1, 2, 3};
// using seq = to_sequence_v<arr>; // seq is ck::Sequence<1, 2, 3>
template <typename T, const T& Arr>
struct to_sequence_t
{
    private:
    template <std::size_t... Is>
    static auto get_sequence_type(std::index_sequence<Is...>) -> ck::Sequence<Arr[Is]...>;

    // Helper method to handler the unusual .Size() method name in ck::Array.
    static constexpr auto get_size(const auto& arr)
    {
        if constexpr(requires { arr.size(); })
        {
            return arr.size();
        }
        else
        {
            return arr.Size();
        }
    }

    public:
    using value = decltype(get_sequence_type(std::make_index_sequence<get_size(Arr)>{}));
};

template <auto& Arr>
using to_sequence_v = typename to_sequence_t<std::remove_cvref_t<decltype(Arr)>, Arr>::value;

// Wrapper function to make constexpr strings a structural type for NTTP.
template <size_t N>
struct StringLiteral
{
    char data[N];
    constexpr StringLiteral(const char (&str)[N])
    {
        for(size_t i = 0; i < N; ++i)
            data[i] = str[i];
    }

    constexpr bool operator==(const StringLiteral<N>& other) const
    {
        for(size_t i = 0; i < N; ++i)
        {
            if(data[i] != other.data[i])
            {
                return false;
            }
        }
        return true;
    }
};

// This is a C++17 deduction guide. It allows the compiler to automatically
// deduce the template argument `N` for `StringLiteral` from a string literal
// constructor argument. For example, you can write `StringLiteral s{"foo"};`
// instead of `StringLiteral<4> s{"foo"};`.
template <size_t N>
StringLiteral(const char (&)[N]) -> StringLiteral<N>;

// Helper to provide a readable error for unsupported enum values.
// The compiler will print the name of this struct in the error message, so
// the name of the enum value will appear instead of just its integer value.
template <auto T>
struct UnsupportedEnumValue
{
};

} // namespace ck_tile::builder
