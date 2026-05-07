// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <stdio.h>
#include <tuple>
#include <utility>

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/print.hpp"
#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile {
template <auto... val>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}
template <typename... type>
[[deprecated("Help function to print value")]] inline constexpr void CK_PRINT()
{
}

template <typename DataType_, typename StaticTileDistribution_>
struct static_distributed_tensor;

template <typename T_, index_t N_>
struct thread_buffer;

// Usage example: CK_PRINTF<float>{}(tensor);
template <typename ConvertTo = void,
          typename FMT       = str_literal<>,
          typename PREFIX    = str_literal<>,
          typename SUFFIX    = str_literal<>>
struct CK_PRINTF;
template <typename ConvertTo, char... FMTChars, char... PREFIXChars, char... SUFFIXChars>
struct CK_PRINTF<ConvertTo,
                 str_literal<FMTChars...>,
                 str_literal<PREFIXChars...>,
                 str_literal<SUFFIXChars...>>
{
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr auto default_format_and_type()
    {
        if constexpr(std::is_same_v<T, float>)
            return std::make_tuple(make_str_literal("%8.3f"), T{});
        else if constexpr(std::is_same_v<T, int>)
            return std::make_tuple(make_str_literal("%5d"), T{});
        else if constexpr(std::is_same_v<T, unsigned int>)
            return std::make_tuple(make_str_literal("%5u"), T{});
        else if constexpr(sizeof(T) == 1)
            return std::make_tuple(make_str_literal("0x%02hhx"), uint8_t{});
        else if constexpr(sizeof(T) == 2)
            return std::make_tuple(make_str_literal("0x%04hx"), uint16_t{});
        else if constexpr(sizeof(T) == 4)
            return std::make_tuple(make_str_literal("0x%08x"), uint32_t{});
        else
            static_assert(false, "Unsupported type");
    }
    template <typename T>
    using default_format_t =
        std::remove_reference_t<decltype(std::get<0>(default_format_and_type<T>()))>;
    template <typename T>
    using default_type_t =
        std::remove_reference_t<decltype(std::get<1>(default_format_and_type<T>()))>;

    CK_TILE_HOST_DEVICE static constexpr auto get_prefix()
    {
        constexpr auto fmt_tid = make_str_literal("tid %03d: [%02d] ");
        if constexpr(sizeof...(PREFIXChars) == 0)
            return fmt_tid;
        else
            return fmt_tid + make_str_literal(" ") + str_literal<PREFIXChars...>{};
    }
    CK_TILE_HOST_DEVICE static constexpr auto get_suffix()
    {
        constexpr auto lf = make_str_literal("\n");
        if constexpr(sizeof...(SUFFIXChars) == 0)
            return lf;
        else
            return str_literal<SUFFIXChars...>{} + lf;
    }

    template <typename T, index_t N, typename Y, index_t... Is, typename... Args>
    CK_TILE_HOST_DEVICE void impl(const thread_buffer<T, N>& buf,
                                  std::integer_sequence<index_t, Is...>,
                                  Args&&... args) const
    {
        using FMT1 = std::
            conditional_t<sizeof...(FMTChars) == 0, default_format_t<Y>, str_literal<FMTChars...>>;
        constexpr auto fmt_v      = FMT1::template duplicate_n<N>(make_str_literal(" "));
        constexpr auto fmt_wrap_v = get_prefix() + fmt_v + get_suffix();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
        printf(fmt_wrap_v.data,
               get_thread_id(),
               N,
               args...,
               bit_cast<default_type_t<Y>>(type_convert<Y>(buf[Is]))...);
#pragma clang diagnostic pop
    }

    template <typename T, index_t N, typename... Args>
    CK_TILE_HOST_DEVICE void operator()(const thread_buffer<T, N>& buf, Args&&... args) const
    {
        using ConvertTo_ = std::conditional_t<std::is_same_v<ConvertTo, void>, T, ConvertTo>;
        impl<T, N, ConvertTo_>(
            buf, std::make_integer_sequence<index_t, N>{}, std::forward<Args>(args)...);
    }

    template <typename... TS, typename... Args>
    CK_TILE_HOST_DEVICE void operator()(const static_distributed_tensor<TS...>& tensor,
                                        Args&&... args) const
    {
        return operator()(tensor.get_thread_buffer(), std::forward<Args>(args)...);
    }
};

template <typename T>
CK_TILE_HOST_DEVICE void print_warp0(T&& x)
{
    if(get_thread_id() < get_warp_size())
        print(std::forward<T>(x));
}
template <typename... Ts>
struct CK_PRINTF_WARP0 : public CK_PRINTF<Ts...>
{
    using base_t = CK_PRINTF<Ts...>;

    template <typename T, typename... Args>
    CK_TILE_HOST_DEVICE void operator()(const T& buf, Args&&... args) const
    {
        if(get_thread_id() < get_warp_size())
            base_t::operator()(buf, std::forward<Args>(args)...);
    }
};

/*
 * RAII struct which inserts start/end markers into the generated assembly.
 *
 * Usage:
 *   - Create an `AsmScopeMarker` object at the beginning of a scope or code block.
 *   - Its constructor will emit a "CK_ASM_SCOPE_START" marker into the assembly.
 *   - When the object goes out of scope (end of block, return, exception, etc.),
 *     the destructor will emit a "CK_ASM_SCOPE_END" marker.
 *
 * Example:
 *   {
 *       [[maybe_unused]] AsmScopeMarker marker;   // Emits CK_ASM_SCOPE_START
 *       // ... code you want to delimit in assembly ...
 *   } // marker goes out of scope → Emits CK_ASM_SCOPE_END
 *
 */
struct AsmScopeMarker
{
    // in some future version of clang we might be able to use string_view to customize
    CK_TILE_HOST_DEVICE AsmScopeMarker() { asm volatile(";;# CK_ASM_SCOPE_START"); }
    CK_TILE_HOST_DEVICE ~AsmScopeMarker() { asm volatile(";;# CK_ASM_SCOPE_END"); }
};

} // namespace ck_tile
