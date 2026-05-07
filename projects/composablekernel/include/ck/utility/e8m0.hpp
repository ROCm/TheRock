// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef CK_CODE_GEN_RTC
#include "ck/utility/type.hpp"

namespace ck {

/**
 * @brief Unsigned representation of a conventional biased Float32 exponent.
 *
 * bias = 127;
 *
 * E8M0_1   = 0b01111111; => 2^(127-127) = 1
 * E8M0_2   = 0b10000000; => 2^(128-127) = 2^1 = 2
 * E8M0_3   = 0b10000010; => 2^(130-127) = 2^3 = 8
 * E8M0_135 = 0b10000111; => 2^(135-127) = 2^8 = 256
 * E8M0_142 = 0b10001110; => 2^(142-127) = 2^15 = 32768
 * E8M0_MIN = 0b00000000; => 2^-127
 * E8M0_MAX = 0b11111110; => 2^127
 * E8M0_NAN = 0b11111111; => NaN
 */
struct e8m0_bexp_t
{
    using type = uint8_t;
    type data;

    constexpr static type bias     = 127;
    constexpr static type nan_mask = 0xFF;

    __host__ __device__ constexpr e8m0_bexp_t() : data{type{}} {}
    __host__ __device__ constexpr e8m0_bexp_t(type init) : data{init} {}
    __host__ __device__ constexpr e8m0_bexp_t(int init) : data{static_cast<type>(init & nan_mask)}
    {
    }
    __host__ __device__ explicit constexpr e8m0_bexp_t(float scale)
        : data{static_cast<type>((bit_cast<uint32_t>(scale) & (nan_mask << 23)) >> 23)}
    {
    }

    __host__ __device__ explicit constexpr operator float() const
    {
        if(data == nan_mask || data == 0)
        {
            uint32_t bits = data << 1;
            bits |= 1;
            bits <<= 22;
            return bit_cast<float>(bits);
        }
        else
        {
            uint32_t bits = data << 23;
            return bit_cast<float>(bits);
        }
    }

    __host__ __device__ constexpr bool operator==(const e8m0_bexp_t& other) const
    {
        // strict IEEE compliance for NaN
        return data == other.data && data != nan_mask;
    }

    __host__ __device__ constexpr bool is_nan() const { return data == nan_mask; }
};

namespace utils {

template <typename T>
__host__ __device__ inline constexpr int32_t get_exponent_value(T x);

template <>
__host__ __device__ inline constexpr int32_t get_exponent_value<e8m0_bexp_t>(e8m0_bexp_t x)
{
    return x.data;
}

} // namespace utils

} // namespace ck
#endif
