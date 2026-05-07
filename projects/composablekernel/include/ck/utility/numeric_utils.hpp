// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck/utility/data_type.hpp"

namespace ck {

template <typename T>
struct NumericUtils
{
};

#ifndef CK_CODE_GEN_RTC
template <>
struct NumericUtils<e8m0_bexp_t>
{
    static constexpr int exp  = 8;
    static constexpr int mant = 0;
    static constexpr int bias = 127;

    static constexpr int unbiased_exp_min = -127;
    static constexpr int unbiased_exp_max = 127;
    static constexpr int biased_exp_min   = 0;
    static constexpr int biased_exp_max   = 254;

    using bitwise_type = uint8_t;
};
#endif

template <>
struct NumericUtils<float>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 23;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    static constexpr bool has_inf       = true;
    using bitwise_type                  = uint32_t;
};

template <>
struct NumericUtils<ck::tf32_t>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 10;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    static constexpr bool has_inf       = true;
    using bitwise_type                  = uint32_t;
};

template <>
struct NumericUtils<half_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint32_t Inf       = 0x7C00;
    static constexpr uint32_t NegInf    = 0xFC00;
    static constexpr uint32_t NaN       = 0x7C01;
    static constexpr uint32_t Neg0      = 0x8000;
    static constexpr bool has_inf       = true;
    using bitwise_type                  = uint16_t;
};

template <>
struct NumericUtils<bhalf_t>
{
    static constexpr int exp  = 8;
    static constexpr int mant = 7;
    static constexpr int bias = 128; // negative zero nan mode
    // static constexpr int bias = 127; // ieee mode
};

template <>
struct NumericUtils<f8_fnuz_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 8; // negative zero nan mode
    // static constexpr int bias = 7; // ieee mode
    static constexpr bool has_inf = false;
};

template <>
struct NumericUtils<bf8_fnuz_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 16; // negative zero nan mode
    // static constexpr int bias = 15; // ieee mode
    static constexpr bool has_inf = false;
};
template <>
struct NumericUtils<f8_ocp_t>
{
    static constexpr int exp  = 4;
    static constexpr int mant = 3;
    static constexpr int bias = 7;
};

template <>
struct NumericUtils<bf8_ocp_t>
{
    static constexpr int exp  = 5;
    static constexpr int mant = 2;
    static constexpr int bias = 15;
};

template <>
struct NumericUtils<f4_t>
{
    static constexpr int exp           = 2;
    static constexpr int mant          = 1;
    static constexpr int bias          = 1;
    static constexpr uint32_t sr_shift = 10;

    static constexpr int unbiased_exp_min = 0;
    static constexpr int unbiased_exp_max = 2;
    static constexpr int biased_exp_min   = 1;
    static constexpr int biased_exp_max   = 3;

    static constexpr uint8_t positive_zero_mask = 0b0000;
    static constexpr uint8_t negative_zero_mask = 0b1000;

    static constexpr uint8_t one_mask      = 0b0010;
    static constexpr uint8_t set_sign_mask = 0b0111;

    static constexpr uint8_t data_max_positive_normal_mask = 0b0111;
    static constexpr uint8_t data_max_negative_normal_mask = 0b1111;

    static constexpr uint8_t data_max_positive_subnormal_mask = 0b0001;
    static constexpr uint8_t data_max_negative_subnormal_mask = 0b1001;

    static constexpr bool has_inf = false;

    using bitwise_type = uint8_t;
};

template <>
struct NumericUtils<f6_t>
{
    static constexpr int exp           = 2;
    static constexpr int mant          = 3;
    static constexpr int bias          = 1;
    static constexpr uint32_t sr_shift = 12;

    static constexpr int unbiased_exp_min = 0;
    static constexpr int unbiased_exp_max = 2;
    static constexpr int biased_exp_min   = 1;
    static constexpr int biased_exp_max   = 3;

    static constexpr uint8_t positive_zero_mask = 0b000000;
    static constexpr uint8_t negative_zero_mask = 0b100000;

    static constexpr uint8_t set_sign_mask = 0b011111;

    static constexpr uint8_t data_max_positive_normal_mask = 0b011111;
    static constexpr uint8_t data_max_negative_normal_mask = 0b111111;

    static constexpr uint8_t data_max_positive_subnormal_mask = 0b000111;
    static constexpr uint8_t data_max_negative_subnormal_mask = 0b100111;

    static constexpr bool has_inf  = false;
    static constexpr bool has_nan  = false;
    static constexpr bool has_zero = true;

    using bitwise_type = uint8_t;
};

template <>
struct NumericUtils<bf6_t>
{
    static constexpr int exp           = 3;
    static constexpr int mant          = 2;
    static constexpr int bias          = 3;
    static constexpr uint32_t sr_shift = 11;

    static constexpr int unbiased_exp_min = -2;
    static constexpr int unbiased_exp_max = 4;
    static constexpr int biased_exp_min   = 1;
    static constexpr int biased_exp_max   = 7;

    static constexpr uint8_t positive_zero_mask = 0b000000;
    static constexpr uint8_t negative_zero_mask = 0b100000;

    static constexpr uint8_t set_sign_mask = 0b011111;

    static constexpr uint8_t data_max_positive_normal_mask = 0b011111;
    static constexpr uint8_t data_max_negative_normal_mask = 0b111111;

    static constexpr uint8_t data_max_positive_subnormal_mask = 0b000011;
    static constexpr uint8_t data_max_negative_subnormal_mask = 0b100011;

    static constexpr bool has_inf  = false;
    static constexpr bool has_nan  = false;
    static constexpr bool has_zero = true;

    using bitwise_type = uint8_t;
};
} // namespace ck
