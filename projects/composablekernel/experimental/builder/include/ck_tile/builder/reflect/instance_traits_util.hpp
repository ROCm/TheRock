// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Utility functions and helpers for instance_traits.hpp
// Contains helper functions to convert types, enums, and sequences to string representations.
// The helper function are consteval so that unknown cases cause compile-time errors.

#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <iostream>
#include <limits.h>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/pipeline_enum.hpp"
#include "ck/utility/scheduler_enum.hpp"
#include "ck/utility/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"

namespace ck_tile::reflect::detail {

// Implementation detail for type name mapping
// This is the single source of truth for supported data types that
// returns an empty string to indicate an unsupported type.
namespace impl {
template <typename T>
consteval std::string_view type_name_impl()
{
    if constexpr(std::is_same_v<T, ck::half_t> || std::is_same_v<T, ck_tile::half_t>)
        return "fp16";
    else if constexpr(std::is_same_v<T, float>)
        return "fp32";
    else if constexpr(std::is_same_v<T, ck::tf32_t>)
        return "tf32";
    else if constexpr(std::is_same_v<T, double>)
        return "fp64";
    else if constexpr(std::is_same_v<T, int8_t>)
        return "s8";
    else if constexpr(std::is_same_v<T, int32_t>)
        return "s32";
    else if constexpr(std::is_same_v<T, ck::bhalf_t> || std::is_same_v<T, ck_tile::bf16_t>)
        return "bf16";
    else if constexpr(std::is_same_v<T, ck::f8_t> || std::is_same_v<T, ck_tile::fp8_t>)
        return "fp8";
    else if constexpr(std::is_same_v<T, ck::bf8_t> || std::is_same_v<T, ck_tile::bf8_t>)
        return "bf8";
    else
        return std::string_view{}; // Return empty for supported types
}
} // namespace impl

// Convert data types to string names
// Fails at compile time for unsupported types
template <typename T>
consteval std::string_view type_name()
{
    constexpr auto name = impl::type_name_impl<T>();
    static_assert(!name.empty(), "Unsupported data type");
    return name;
}

// Concept that checks if a type is a valid data type
// Uses the impl directly to avoid triggering static_assert during concept evaluation
template <typename T>
concept IsDataType = !impl::type_name_impl<T>().empty();

// Concept that checks valid layout types
template <typename T>
concept IsLayoutType = (std::is_base_of_v<ck_tile::tensor_layout::BaseTensorLayout, T> ||
                        std::is_base_of_v<ck::tensor_layout::BaseTensorLayout, T>) &&
                       requires {
                           { T::name } -> std::convertible_to<std::string_view>;
                       };

// Convert layout types to string names
template <IsLayoutType T>
constexpr std::string_view layout_name()
{
    return T::name;
}

// Convert element-wise operation types to string names
template <typename T>
constexpr std::string_view elementwise_op_name()
{
    if constexpr(requires {
                     { T::name } -> std::convertible_to<std::string_view>;
                 })
        return T::name;
    else
        static_assert(false, "Elementwise operation is missing name attribute");
}

// Convert ConvolutionForwardSpecialization enum to string
constexpr std::string_view
conv_fwd_spec_name(ck::tensor_operation::device::ConvolutionForwardSpecialization spec)
{
    using enum ck::tensor_operation::device::ConvolutionForwardSpecialization;
    switch(spec)
    {
    case Default: return "Default";
    case Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    case Filter1x1Pad0: return "Filter1x1Pad0";
    case Filter3x3: return "Filter3x3";
    case OddC: return "OddC";
    }
}

// Convert ConvolutionBackwardWeightSpecialization enum to string
constexpr std::string_view conv_bwd_weight_spec_name(
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization spec)
{
    using enum ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization;
    switch(spec)
    {
    case Default: return "Default";
    case Filter1x1Stride1Pad0: return "Filter1x1Stride1Pad0";
    case Filter1x1Pad0: return "Filter1x1Pad0";
    case OddC: return "OddC";
    }
}

// Convert GemmSpecialization enum to string
constexpr std::string_view gemm_spec_name(ck::tensor_operation::device::GemmSpecialization spec)
{
    using enum ck::tensor_operation::device::GemmSpecialization;
    switch(spec)
    {
    case Default: return "Default";
    case MPadding: return "MPadding";
    case NPadding: return "NPadding";
    case KPadding: return "KPadding";
    case MNPadding: return "MNPadding";
    case MKPadding: return "MKPadding";
    case NKPadding: return "NKPadding";
    case MNKPadding: return "MNKPadding";
    case OPadding: return "OPadding";
    case MOPadding: return "MOPadding";
    case NOPadding: return "NOPadding";
    case KOPadding: return "KOPadding";
    case MNOPadding: return "MNOPadding";
    case MKOPadding: return "MKOPadding";
    case NKOPadding: return "NKOPadding";
    case MNKOPadding: return "MNKOPadding";
    }
}

// Convert BlockGemmPipelineScheduler enum to string
constexpr std::string_view pipeline_scheduler_name(ck::BlockGemmPipelineScheduler sched)
{
    using enum ck::BlockGemmPipelineScheduler;
    switch(sched)
    {
    case Intrawave: return "Intrawave";
    case Interwave: return "Interwave";
    }
}

constexpr std::string_view pipeline_scheduler_name(ck_tile::GemmPipelineScheduler sched)
{
    using enum ck_tile::GemmPipelineScheduler;
    switch(sched)
    {
    case Default: return "Default";
    case Intrawave: return "Intrawave";
    case Interwave: return "Interwave";
    }
}

// Convert BlockGemmPipelineVersion enum to string
constexpr std::string_view pipeline_version_name(ck::BlockGemmPipelineVersion ver)
{
    using enum ck::BlockGemmPipelineVersion;
    switch(ver)
    {
    case v1: return "v1";
    case v2: return "v2";
    case v3: return "v3";
    case v4: return "v4";
    case v5: return "v5";
    }
}

// Convert PipelineVersion enum to string (for Wmma kernels)
constexpr std::string_view pipeline_version_name(ck::PipelineVersion ver)
{
    using enum ck::PipelineVersion;
    switch(ver)
    {
    case v1: return "v1";
    case v2: return "v2";
    case v4: return "v4";
    case weight_only: return "weight_only";
    }
}

// Convert LoopScheduler enum to string
constexpr std::string_view loop_scheduler_name(ck::LoopScheduler sched)
{
    using enum ck::LoopScheduler;
    switch(sched)
    {
    case Default: return "Default";
    case Interwave: return "Interwave";
    }
}

// Convert TailNumber enum to string
constexpr std::string_view tail_number_name(ck_tile::TailNumber tail_num)
{
    using enum ck_tile::TailNumber;
    switch(tail_num)
    {
    case Odd: return "Odd";
    case Even: return "Even";
    case One: return "One";
    case Two: return "Two";
    case Three: return "Three";
    case Four: return "Four";
    case Five: return "Five";
    case Six: return "Six";
    case Seven: return "Seven";
    case Empty: return "Empty";
    case Full: return "Full";
    }
}

// Convert std::array to string
template <typename T, std::size_t N>
inline std::string array_to_string(const std::array<T, N>& arr)
{
    std::ostringstream oss;
    oss << "Seq(";
    for(std::size_t i = 0; i < arr.size(); ++i)
    {
        if(i > 0)
            oss << ",";
        oss << arr[i];
    }
    oss << ")";
    return oss.str();
}

// Metaprogramming helper to convert ck::Sequence to constexpr std::array
template <typename Seq>
struct SequenceToArray;

template <ck::index_t... Is>
struct SequenceToArray<ck::Sequence<Is...>>
{
    static constexpr std::array<int, sizeof...(Is)> value = {static_cast<int>(Is)...};
};

namespace detail {
// Generic helper to build list-like strings (Tuple, Seq, etc.)
//
// Example output: "Seq(1,2,3)"
//
// prefix: The list-like container name (e.g. "Tuple" or "Seq")
// converter_fn: A callable that converts each element to a string representation
// For types: converter_fn should be a template lambda like []<typename U>() { return
// type_name<U>(); } For values: converter_fn should be a regular lambda like [](auto value) {
// return std::to_string(value); }
template <typename ConverterFn, typename... Elements>
constexpr std::string build_list_string(std::string_view prefix, const ConverterFn& converter_fn)
{
    if constexpr(sizeof...(Elements) == 0)
    {
        return std::string(prefix) + "()";
    }
    else
    {
        std::string result = std::string(prefix) + "(";
        std::size_t index  = 0;
        ((result +=
          (index++ > 0 ? "," : "") + std::string(converter_fn.template operator()<Elements>())),
         ...);
        result += ")";
        return result;
    }
}

// Overload for value-based lists (sequences)
template <typename ConverterFn, auto... Values>
constexpr std::string build_list_string_values(std::string_view prefix,
                                               const ConverterFn& converter_fn)
{
    if constexpr(sizeof...(Values) == 0)
    {
        return std::string(prefix) + "()";
    }
    else
    {
        std::string result = std::string(prefix) + "(";
        std::size_t index  = 0;
        ((result += (index++ > 0 ? "," : "") + converter_fn(Values)), ...);
        result += ")";
        return result;
    }
}
} // namespace detail

// Convert ck::Sequence to string representation
// Converts a ck::Sequence type to a string in the format "Seq(v1,v2,...,vn)"
// where each value is converted using std::to_string.
//
// Template parameter:
//   T: Must be a ck::Sequence<...> type
//
// Constraints:
//   - Sequence elements must support std::to_string (typically ck::index_t)
//
// Examples:
//   sequence_name<ck::Sequence<>>()           returns "Seq()"
//   sequence_name<ck::Sequence<42>>()         returns "Seq(42)"
//   sequence_name<ck::Sequence<1,2,3>>()      returns "Seq(1,2,3)"
//   sequence_name<ck::Sequence<256,128,64>>() returns "Seq(256,128,64)"
template <typename T>
    requires requires { []<ck::index_t... Is>(ck::Sequence<Is...>*) {}(static_cast<T*>(nullptr)); }
constexpr std::string sequence_name()
{
    return []<ck::index_t... Is>(ck::Sequence<Is...>*) constexpr {
        auto to_string_fn = [](auto value) { return std::to_string(value); };
        return detail::build_list_string_values<decltype(to_string_fn), Is...>("Seq", to_string_fn);
    }(static_cast<T*>(nullptr));
}

// Convert ck::Tuple to string representation
// Converts a ck::Tuple type to a string in the format "Tuple(e1,e2,...,en)"
// where each element is converted based on its type (layout names or data type names).
//
// Template parameter:
//   T: Must be a ck::Tuple<...> type
//
// Constraints:
//   - Empty tuples are supported and return "EmptyTuple"
//   - All tuple elements must be homogeneous: either all layouts (IsLayoutType) or all data types
//   (IsDataType)
//   - Mixed layouts and data types in the same tuple will cause a compile-time error
//
// Examples:
//   tuple_name<ck::Tuple<>>()                                    returns "EmptyTuple"
//   tuple_name<ck::Tuple<ck::tensor_layout::gemm::RowMajor>>()  returns "Tuple(RowMajor)"
//   tuple_name<ck::Tuple<NCHW,NHWC>>()                          returns "Tuple(NCHW,NHWC)"
//   tuple_name<ck::Tuple<ck::half_t>>()                         returns "Tuple(fp16)"
//   tuple_name<ck::Tuple<ck::half_t,float,double>>()            returns "Tuple(fp16,fp32,fp64)"
template <typename T>
    requires requires { []<typename... Ts>(ck::Tuple<Ts...>*) {}(static_cast<T*>(nullptr)); }
constexpr std::string tuple_name()
{
    return []<typename... Ts>(ck::Tuple<Ts...>*) constexpr {
        if constexpr(sizeof...(Ts) == 0)
        {
            return std::string("EmptyTuple");
        }
        else if constexpr((IsLayoutType<Ts> && ...))
        {
            // Lambda wrapper for layout_name
            auto layout_name_fn = []<typename U>() { return layout_name<U>(); };
            return detail::build_list_string<decltype(layout_name_fn), Ts...>("Tuple",
                                                                              layout_name_fn);
        }
        else if constexpr((IsDataType<Ts> && ...))
        {
            // Lambda wrapper for type_name
            auto type_name_fn = []<typename U>() { return type_name<U>(); };
            return detail::build_list_string<decltype(type_name_fn), Ts...>("Tuple", type_name_fn);
        }
        else
        {
            static_assert((IsLayoutType<Ts> && ...) || (IsDataType<Ts> && ...),
                          "Tuple elements must be all layouts or all data types, not mixed");
            return std::string{}; // unreachable
        }
    }(static_cast<T*>(nullptr));
}

template <typename T>
    requires requires { []<typename... Ts>(ck_tile::tuple<Ts...>*) {}(static_cast<T*>(nullptr)); }
constexpr std::string tuple_name()
{
    return []<typename... Ts>(ck_tile::tuple<Ts...>*) constexpr {
        if constexpr(sizeof...(Ts) == 0)
        {
            return std::string("EmptyTuple");
        }
        else if constexpr((IsLayoutType<Ts> && ...))
        {
            // Lambda wrapper for layout_name
            auto layout_name_fn = []<typename U>() { return layout_name<U>(); };
            return detail::build_list_string<decltype(layout_name_fn), Ts...>("tuple",
                                                                              layout_name_fn);
        }
        else if constexpr((IsDataType<Ts> && ...))
        {
            // Lambda wrapper for type_name
            auto type_name_fn = []<typename U>() { return type_name<U>(); };
            return detail::build_list_string<decltype(type_name_fn), Ts...>("tuple", type_name_fn);
        }
        else
        {
            static_assert((IsLayoutType<Ts> && ...) || (IsDataType<Ts> && ...),
                          "tuple elements must be all layouts or all data types, not mixed");
            return std::string{}; // unreachable
        }
    }(static_cast<T*>(nullptr));
}

// Concept to check if a type is a ck::Tuple
template <typename T>
concept IsCkTuple =
    requires { []<typename... Ts>(ck::Tuple<Ts...>*) {}(static_cast<T*>(nullptr)); };

// Concept to check if a type is a ck_tile::tuple
template <typename T>
concept IsCkTileTuple =
    requires { []<typename... Ts>(ck_tile::tuple<Ts...>*) {}(static_cast<T*>(nullptr)); };

// Deduces whether to use tuple_name or type_name
// Handles both scalar data types and ck::Tuple types
template <typename T>
constexpr std::string type_or_type_tuple_name()
{
    if constexpr(IsCkTuple<T> || IsCkTileTuple<T>)
    {
        return tuple_name<T>();
    }
    else
    {
        return std::string(type_name<T>());
    }
}

/// @brief Makes a case insensitive comparison of two string views.
/// @param a First string view
/// @param b Second string view
/// @return Whether two string views a equal case insensitive
constexpr bool case_insensitive_equal(std::string_view a, std::string_view b)
{
    if(a.size() != b.size())
        return false;

    for(size_t i = 0; i < a.size(); ++i)
    {
        char c1 = a[i];
        char c2 = b[i];

        // Convert to lowercase for comparison
        if(c1 >= 'A' && c1 <= 'Z')
            c1 += 32;
        if(c2 >= 'A' && c2 <= 'Z')
            c2 += 32;

        if(c1 != c2)
            return false;
    }
    return true;
}

} // namespace ck_tile::reflect::detail
