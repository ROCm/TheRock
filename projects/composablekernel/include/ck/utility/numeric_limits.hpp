// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck/utility/data_type.hpp"

namespace ck {

#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
template <typename T>
struct NumericLimits;

template <>
struct NumericLimits<int32_t>
{
    __host__ __device__ static constexpr int32_t Lowest() noexcept { return -2147483647 - 1; }

    __host__ __device__ static constexpr int32_t Min() noexcept { return -2147483647 - 1; }

    __host__ __device__ static constexpr int32_t Max() noexcept { return 2147483647; }

    __host__ __device__ static constexpr int32_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int32_t QuietNaN() { return 0; }
};
template <>
struct NumericLimits<int16_t>
{
    __host__ __device__ static constexpr int16_t Lowest() noexcept { return -32768; }

    __host__ __device__ static constexpr int16_t Min() noexcept { return -32768; }

    __host__ __device__ static constexpr int16_t Max() noexcept { return 32767; }

    __host__ __device__ static constexpr int16_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int16_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<int8_t>
{
    __host__ __device__ static constexpr int8_t Lowest() noexcept { return -128; }

    __host__ __device__ static constexpr int8_t Min() noexcept { return -128; }

    __host__ __device__ static constexpr int8_t Max() noexcept { return 127; }

    __host__ __device__ static constexpr int8_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr int8_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<uint32_t>
{
    __host__ __device__ static constexpr uint32_t Lowest() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t Min() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t Max() noexcept { return 4294967295U; }

    __host__ __device__ static constexpr uint32_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr uint32_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<uint16_t>
{
    __host__ __device__ static constexpr uint16_t Lowest() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t Min() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t Max() noexcept { return 65535U; }

    __host__ __device__ static constexpr uint16_t Infinity() noexcept { return 0; }

    __host__ __device__ static constexpr uint16_t QuietNaN() { return 0; }
};

template <>
struct NumericLimits<float>
{
    static constexpr unsigned int binary_min    = 0x00800000;
    static constexpr unsigned int binary_max    = 0x7F7FFFFF;
    static constexpr unsigned int binary_lowest = 0xFF7FFFFF;
    static constexpr unsigned int binary_qnan   = 0xFFC00001;
    static constexpr unsigned int binary_inf    = 0x7F800000;

    __host__ __device__ static constexpr float Min() { return bit_cast<float>(binary_min); }

    __host__ __device__ static constexpr float Max() { return bit_cast<float>(binary_max); }

    __host__ __device__ static constexpr float Lowest() { return bit_cast<float>(binary_lowest); }

    __host__ __device__ static constexpr float QuietNaN() { return bit_cast<float>(binary_qnan); }

    __host__ __device__ static constexpr float Infinity() { return bit_cast<float>(binary_inf); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

template <>
struct NumericLimits<f8_fnuz_t>
{
    // negative zero nan mode with exp bias = 8
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 7
    // static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    // static constexpr uint8_t binary_max    = 0x77; // 0b01110111
    // static constexpr uint8_t binary_lowest = 0xF7; // 0b11110111
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=0

    __host__ __device__ static constexpr f8_fnuz_t Min() { return f8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr f8_fnuz_t Max() { return f8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr f8_fnuz_t Lowest() { return f8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr f8_fnuz_t QuietNaN() { return f8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_fnuz_t>
{
    // negative zero nan mode with exp bias = 16
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 15
    // static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    // static constexpr uint8_t binary_max    = 0x7B; // 0b01111011
    // static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=

    __host__ __device__ static constexpr bf8_fnuz_t Min() { return bf8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr bf8_fnuz_t Max() { return bf8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr bf8_fnuz_t Lowest() { return bf8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_fnuz_t QuietNaN() { return bf8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<f8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr f8_ocp_t Min() { return bit_cast<f8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr f8_ocp_t Max() { return bit_cast<f8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr f8_ocp_t Lowest()
    {
        return bit_cast<f8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr f8_ocp_t QuietNaN()
    {
        return bit_cast<f8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<bf8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_ocp_t Min() { return bit_cast<bf8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr bf8_ocp_t Max() { return bit_cast<bf8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr bf8_ocp_t Lowest()
    {
        return bit_cast<bf8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr bf8_ocp_t QuietNaN()
    {
        return bit_cast<bf8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<f4_t>
{
    static constexpr uint8_t binary_min_normal    = 0x2; // 0b0010
    static constexpr uint8_t binary_max_normal    = 0x7; // 0b0111
    static constexpr uint8_t binary_lowest_normal = 0xF; // 0b1111
    static constexpr uint8_t binary_min_subnorm   = 0x1; // 0b0001
    static constexpr uint8_t binary_max_subnorm   = 0x1; // 0b0001

    static constexpr float data_max_normal_number    = 6;
    static constexpr float data_min_subnormal_number = 0.5;

    __host__ __device__ static constexpr f4_t Min() { return f4_t(binary_min_normal); }
    __host__ __device__ static constexpr f4_t Max() { return f4_t(binary_max_normal); }
    __host__ __device__ static constexpr f4_t Lowest() { return f4_t(binary_lowest_normal); }
    __host__ __device__ static constexpr f4_t MinSubnorm() { return f4_t(binary_min_subnorm); }
    __host__ __device__ static constexpr f4_t MaxSubnorm() { return f4_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<f6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x07; // 0b000111

    static constexpr float data_max_normal_number    = 7.5;
    static constexpr float data_min_subnormal_number = 0.125;

    __host__ __device__ static constexpr f6_t Min() { return f6_t(binary_min_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Max() { return f6_t(binary_max_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Lowest()
    {
        return f6_t(binary_lowest_normal & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MinSubnorm()
    {
        return f6_t(binary_min_subnorm & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MaxSubnorm()
    {
        return f6_t(binary_max_subnorm & 0b111111);
    }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<bf6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x03; // 0b000011

    static constexpr float data_max_normal_number    = 28;
    static constexpr float data_min_subnormal_number = 0.0625;

    __host__ __device__ static constexpr bf6_t Min() { return bf6_t(binary_min_normal); }
    __host__ __device__ static constexpr bf6_t Max() { return bf6_t(binary_max_normal); }
    __host__ __device__ static constexpr bf6_t Lowest() { return bf6_t(binary_lowest_normal); }
    __host__ __device__ static constexpr bf6_t MinSubnorm() { return bf6_t(binary_min_subnorm); }
    __host__ __device__ static constexpr bf6_t MaxSubnorm() { return bf6_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

#else
template <typename T>
struct NumericLimits
{
    __host__ __device__ static constexpr T Min() { return std::numeric_limits<T>::min(); }
    __host__ __device__ static constexpr T Max() { return std::numeric_limits<T>::max(); }
    __host__ __device__ static constexpr T Lowest() { return std::numeric_limits<T>::lowest(); }
    __host__ __device__ static constexpr T QuietNaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }
    __host__ __device__ static constexpr T Infinity() { return std::numeric_limits<T>::infinity(); }
};

template <>
struct NumericLimits<half_t>
{
    static constexpr unsigned short binary_min    = 0x0400;
    static constexpr unsigned short binary_max    = 0x7BFF;
    static constexpr unsigned short binary_lowest = 0xFBFF;
    static constexpr unsigned short binary_qnan   = 0x7FFF;

    __host__ __device__ static constexpr half_t Min() { return bit_cast<half_t>(binary_min); }

    __host__ __device__ static constexpr half_t Max() { return bit_cast<half_t>(binary_max); }

    __host__ __device__ static constexpr half_t Lowest() { return bit_cast<half_t>(binary_lowest); }

    __host__ __device__ static constexpr half_t QuietNaN() { return bit_cast<half_t>(binary_qnan); }
};

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
template <>
struct NumericLimits<int4_t>
{
    __host__ __device__ static constexpr int4_t Min() { return int4_t(-8); }

    __host__ __device__ static constexpr int4_t Max() { return int4_t(7); }

    __host__ __device__ static constexpr int4_t Lowest() { return int4_t(-8); }
};
#endif // CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4

template <>
struct NumericLimits<f8_fnuz_t>
{
    // negative zero nan mode with exp bias = 8
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 7
    // static constexpr uint8_t binary_min    = 0x08; // 0b00001000
    // static constexpr uint8_t binary_max    = 0x77; // 0b01110111
    // static constexpr uint8_t binary_lowest = 0xF7; // 0b11110111
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=0

    __host__ __device__ static constexpr f8_fnuz_t Min() { return f8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr f8_fnuz_t Max() { return f8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr f8_fnuz_t Lowest() { return f8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr f8_fnuz_t QuietNaN() { return f8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<bf8_fnuz_t>
{
    // negative zero nan mode with exp bias = 16
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    static constexpr uint8_t binary_max    = 0x7F; // 0b01111111
    static constexpr uint8_t binary_lowest = 0xFF; // 0b11111111
    static constexpr uint8_t binary_qnan   = 0x80; // 0b10000000
    // ieee mode with exp bias = 15
    // static constexpr uint8_t binary_min    = 0x04; // 0b00000100
    // static constexpr uint8_t binary_max    = 0x7B; // 0b01111011
    // static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011
    // static constexpr uint8_t binary_qnan   = 0x79; // any sign, exp=1111, mant!=

    __host__ __device__ static constexpr bf8_fnuz_t Min() { return bf8_fnuz_t(binary_min); }

    __host__ __device__ static constexpr bf8_fnuz_t Max() { return bf8_fnuz_t(binary_max); }

    __host__ __device__ static constexpr bf8_fnuz_t Lowest() { return bf8_fnuz_t(binary_lowest); }

    __host__ __device__ static constexpr bf8_fnuz_t QuietNaN() { return bf8_fnuz_t(binary_qnan); }
};

template <>
struct NumericLimits<f8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x08; // 0b00001000 = 2^-6
    static constexpr uint8_t binary_max    = 0x7E; // 0b01111110 = 448
    static constexpr uint8_t binary_lowest = 0xFE; // 0b11111110 = -448
    static constexpr uint8_t binary_qnan   = 0x7F; // 0b01111111

    __host__ __device__ static constexpr f8_ocp_t Min() { return bit_cast<f8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr f8_ocp_t Max() { return bit_cast<f8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr f8_ocp_t Lowest()
    {
        return bit_cast<f8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr f8_ocp_t QuietNaN()
    {
        return bit_cast<f8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<bf8_ocp_t>
{
    static constexpr uint8_t binary_min    = 0x04; // 0b00000100 = 2^-14
    static constexpr uint8_t binary_max    = 0x7B; // 0b01111011 = 57344
    static constexpr uint8_t binary_lowest = 0xFB; // 0b11111011 = -57344
    static constexpr uint8_t binary_qnan   = 0x7D; // 0b01111101

    __host__ __device__ static constexpr bf8_ocp_t Min() { return bit_cast<bf8_ocp_t>(binary_min); }

    __host__ __device__ static constexpr bf8_ocp_t Max() { return bit_cast<bf8_ocp_t>(binary_max); }

    __host__ __device__ static constexpr bf8_ocp_t Lowest()
    {
        return bit_cast<bf8_ocp_t>(binary_lowest);
    }

    __host__ __device__ static constexpr bf8_ocp_t QuietNaN()
    {
        return bit_cast<bf8_ocp_t>(binary_qnan);
    }
};

template <>
struct NumericLimits<f4_t>
{
    static constexpr uint8_t binary_min_normal    = 0x2; // 0b0010
    static constexpr uint8_t binary_max_normal    = 0x7; // 0b0111
    static constexpr uint8_t binary_lowest_normal = 0xF; // 0b1111
    static constexpr uint8_t binary_min_subnorm   = 0x1; // 0b0001
    static constexpr uint8_t binary_max_subnorm   = 0x1; // 0b0001

    static constexpr float data_max_normal_number    = 6;
    static constexpr float data_min_subnormal_number = 0.5;

    __host__ __device__ static constexpr f4_t Min() { return f4_t(binary_min_normal); }
    __host__ __device__ static constexpr f4_t Max() { return f4_t(binary_max_normal); }
    __host__ __device__ static constexpr f4_t Lowest() { return f4_t(binary_lowest_normal); }
    __host__ __device__ static constexpr f4_t MinSubnorm() { return f4_t(binary_min_subnorm); }
    __host__ __device__ static constexpr f4_t MaxSubnorm() { return f4_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<f6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x07; // 0b000111

    static constexpr float data_max_normal_number    = 7.5;
    static constexpr float data_min_subnormal_number = 0.125;

    __host__ __device__ static constexpr f6_t Min() { return f6_t(binary_min_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Max() { return f6_t(binary_max_normal & 0b111111); }
    __host__ __device__ static constexpr f6_t Lowest()
    {
        return f6_t(binary_lowest_normal & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MinSubnorm()
    {
        return f6_t(binary_min_subnorm & 0b111111);
    }
    __host__ __device__ static constexpr f6_t MaxSubnorm()
    {
        return f6_t(binary_max_subnorm & 0b111111);
    }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<bf6_t>
{
    static constexpr uint8_t binary_min_normal    = 0x08; // 0b001000
    static constexpr uint8_t binary_max_normal    = 0x1F; // 0b011111
    static constexpr uint8_t binary_lowest_normal = 0x3F; // 0b111111
    static constexpr uint8_t binary_min_subnorm   = 0x01; // 0b000001
    static constexpr uint8_t binary_max_subnorm   = 0x03; // 0b000011

    static constexpr float data_max_normal_number    = 28;
    static constexpr float data_min_subnormal_number = 0.0625;

    __host__ __device__ static constexpr bf6_t Min() { return bf6_t(binary_min_normal); }
    __host__ __device__ static constexpr bf6_t Max() { return bf6_t(binary_max_normal); }
    __host__ __device__ static constexpr bf6_t Lowest() { return bf6_t(binary_lowest_normal); }
    __host__ __device__ static constexpr bf6_t MinSubnorm() { return bf6_t(binary_min_subnorm); }
    __host__ __device__ static constexpr bf6_t MaxSubnorm() { return bf6_t(binary_max_subnorm); }

    __host__ __device__ static constexpr float DataMaxNorm() { return data_max_normal_number; }
    __host__ __device__ static constexpr float DataMinSubnorm()
    {
        return data_min_subnormal_number;
    }
};

template <>
struct NumericLimits<e8m0_bexp_t>
{
    static constexpr e8m0_bexp_t binary_min  = 0x00; // 0b00000000
    static constexpr e8m0_bexp_t binary_max  = 0xFE; // 0b11111110
    static constexpr e8m0_bexp_t binary_qnan = 0xFF; // 0b11111111
    static constexpr e8m0_bexp_t binary_1    = 0x7F; // 0b01111111
    static constexpr e8m0_bexp_t binary_2    = 0x80; // 0b10000000
    static constexpr e8m0_bexp_t binary_3    = 0x82; // 0b10000010
    static constexpr e8m0_bexp_t binary_135  = 0x87; // 0b10000111
    static constexpr e8m0_bexp_t binary_142  = 0x8E; // 0b10001110

    __host__ __device__ static constexpr e8m0_bexp_t Min() { return e8m0_bexp_t(binary_min); }
    __host__ __device__ static constexpr e8m0_bexp_t Max() { return e8m0_bexp_t(binary_max); }
    __host__ __device__ static constexpr e8m0_bexp_t QuietNaN() { return e8m0_bexp_t(binary_qnan); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_1() { return e8m0_bexp_t(binary_1); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_2() { return e8m0_bexp_t(binary_2); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_3() { return e8m0_bexp_t(binary_3); }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_135()
    {
        return e8m0_bexp_t(binary_135);
    }
    __host__ __device__ static constexpr e8m0_bexp_t Binary_142()
    {
        return e8m0_bexp_t(binary_142);
    }
};
#endif

} // namespace ck
