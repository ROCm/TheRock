// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

namespace ck_tile {

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
    using raw_type = uint8_t;
    using type     = raw_type;

    raw_type data;

    CK_TILE_HOST_DEVICE constexpr e8m0_bexp_t() : data{type{0b11111111}} {}
    CK_TILE_HOST_DEVICE explicit constexpr e8m0_bexp_t(type init) : data{init} {}
    CK_TILE_HOST_DEVICE explicit constexpr e8m0_bexp_t(float scale)
        : e8m0_bexp_t(static_cast<type>(numeric_utils<float>::get_exponent(scale)))
    {
    }
    CK_TILE_HOST_DEVICE constexpr operator type() const { return data; }
    CK_TILE_HOST_DEVICE constexpr raw_type& get() { return data; }
    CK_TILE_HOST_DEVICE constexpr raw_type get() const { return data; }
    CK_TILE_HOST_DEVICE constexpr operator float() const;

    constexpr bool operator==(const e8m0_bexp_t& other) const { return data == other.data; }

    constexpr bool operator!=(const e8m0_bexp_t& other) const { return data != other.data; }
};

using e8m0_t     = e8m0_bexp_t;
using e8m0_raw_t = typename e8m0_t::raw_type;

template <>
struct numeric_traits<e8m0_t>
{
    using bitwise_type = e8m0_raw_t;

    static constexpr int exp        = 8;
    static constexpr int mant       = 0;
    static constexpr int bias       = 127;
    static constexpr int PackedSize = 1;
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<e8m0_t>
{
    static constexpr e8m0_raw_t binary_min = 0b00000000; // 2^-127
    static constexpr e8m0_raw_t binary_max = 0b11111110; // 2^127
    static constexpr e8m0_raw_t binary_nan = 0b11111111;
    CK_TILE_HOST_DEVICE static constexpr e8m0_t min() { return e8m0_t{binary_min}; }
    CK_TILE_HOST_DEVICE static constexpr e8m0_t max() { return e8m0_t{binary_max}; }
    CK_TILE_HOST_DEVICE static constexpr e8m0_t quiet_NaN() { return e8m0_t{binary_nan}; }
    CK_TILE_HOST_DEVICE static constexpr e8m0_t signaling_NaN() { return e8m0_t{binary_nan}; }
    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }

    CK_TILE_HOST_DEVICE static constexpr e8m0_t epsilon() { return signaling_NaN(); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_t round_error() { return signaling_NaN(); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_t zero() { return signaling_NaN(); }
    CK_TILE_HOST_DEVICE static constexpr e8m0_t infinity() { return signaling_NaN(); }
};

CK_TILE_HOST_DEVICE constexpr e8m0_bexp_t::operator float() const
{
    using traits = numeric_traits<float>;
    if(data == numeric<e8m0_t>::binary_nan)
    {
        return std::numeric_limits<float>::signaling_NaN();
    }
    else if(data == 0)
    {
        return std::numeric_limits<float>::min();
    }
    else
    {
        return bit_cast<float>(static_cast<traits::bitwise_type>(data) << traits::mant);
    }
}

} // namespace ck_tile
