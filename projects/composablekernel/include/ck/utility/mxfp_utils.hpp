// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"

#ifdef CK_CODE_GEN_RTC
#define UINT_MAX 4294967295
#endif
namespace ck::utils {

union cvt
{
    float value_float;
    uint32_t value_bitwise;
};

template <typename DTYPE>
inline bool getDataHasInf()
{
    return DTYPE::dataInfo.hasInf;
}

template <typename T>
__host__ __device__ inline bool is_zero(e8m0_bexp_t const scale, T const data);

template <typename T>
__host__ __device__ inline bool is_nan(e8m0_bexp_t const scale, T const data);

template <typename T>
__host__ __device__ inline bool is_inf(e8m0_bexp_t const scale, T const data);

template <typename T>
__host__ __device__ inline constexpr int32_t get_exponent_value(T x)
{
    x >>= NumericUtils<T>::mant;

    x &= ((1 << NumericUtils<T>::exp) - 1);

    return static_cast<int32_t>(x);
}

template <typename T>
__host__ __device__ inline bool is_subnormal(T x)
{
    return get_exponent_value<T>(x) == 0;
}

template <typename T>
__host__ __device__ inline double get_mantissa_value(T x)
{
    double mantissa = is_subnormal<T>(x) ? 0.0f : 1.0f;

    for(uint i = 0; i < NumericUtils<T>::mant; i++)
    {

        mantissa += std::pow(2, -int32_t((NumericUtils<T>::mant - i))) * (x & 0b1);

        x >>= 1;
    }

    return mantissa;
}

template <typename T>
__host__ __device__ inline bool get_data_has_inf()
{
    return NumericUtils<T>::has_inf;
}

template <typename T>
__host__ __device__ float convert_to_float(T data, int scale_exp)
{
    float d_sign =
        std::pow(-1, static_cast<float>(data >> (NumericUtils<T>::exp + NumericUtils<T>::mant)));

    float d_exp;
    if(is_subnormal<T>(data))
        d_exp = std::pow(2, 1 - static_cast<int>(NumericUtils<T>::bias));
    else
        d_exp = std::pow(2, get_exponent_value<T>(data) - static_cast<int>(NumericUtils<T>::bias));
    float d_mant = get_mantissa_value<T>(data);

    float data_value  = d_sign * d_exp * d_mant;
    float scale_value = std::pow(
        2, static_cast<float>((scale_exp - static_cast<int>(NumericUtils<e8m0_bexp_t>::bias))));

    return data_value * scale_value;
}

template <typename T>
__host__ __device__ inline float to_float(e8m0_bexp_t const scale, T const data);

template <typename T>
__host__ __device__ T sat_convert_to_type(float value);

template <typename T>
__host__ __device__ T sat_convert_to_type_sr(float value, uint32_t seed);

template <typename T>
__host__ __device__ inline T convert_to_type(float value)
{
    using bitwise_type = typename NumericUtils<T>::bitwise_type;

    if(std::abs(value) > NumericLimits<T>::Max())
    {
        float max_value = NumericLimits<T>::Max();

        cvt t;

        // cppcheck-suppress redundantAssignment
        t.value_float        = max_value;
        uint32_t max_bitwise = t.value_bitwise;

        // cppcheck-suppress redundantAssignment
        t.value_float = value;
        bitwise_type sign =
            t.value_bitwise >> (NumericUtils<float>::exp + NumericUtils<float>::mant);
        bitwise_type exp =
            ((max_bitwise >> NumericUtils<float>::mant) & NumericUtils<float>::exp_mask) -
            (NumericUtils<float>::bias - NumericUtils<T>::bias);
        bitwise_type mantissa = max_bitwise >> (NumericUtils<float>::mant - NumericUtils<T>::mant);

        uint32_t mant_prev = max_bitwise >> (NumericUtils<float>::mant - NumericUtils<T>::mant);
        mant_prev &= ((1 << NumericUtils<T>::mant) - 1);
        mant_prev--;

        mant_prev <<= (NumericUtils<float>::mant - NumericUtils<T>::mant);
        uint32_t prev_bit =
            ((max_bitwise >> NumericUtils<float>::mant) << NumericUtils<float>::mant) | mant_prev;

        t.value_bitwise = prev_bit;
        float prev_val  = t.value_float;
        float diff      = max_value - prev_val;

        float actual_max = max_value + (diff / 2);

        if(std::abs(value) < actual_max)
        {
            return sign << ((NumericUtils<T>::exp + NumericUtils<T>::mant)) |
                   (exp << NumericUtils<T>::mant) | mantissa;
        }
        else
        {
            if(!get_data_has_inf<T>())
            {

                return (1 << (NumericUtils<T>::mant + NumericUtils<T>::exp)) - 1;
            }
            else
            {
                exp++;
                return sign << ((NumericUtils<T>::exp + NumericUtils<T>::mant)) |
                       (exp << NumericUtils<T>::mant);
            }
        }
    }
    const int mfmt = NumericUtils<float>::mant;
    uint32_t x;
    x = bit_cast<uint32_t>(value);

    uint32_t head, mantissa;
    int32_t exponent, bias;
    uint32_t sign;

    head     = x & NumericUtils<float>::head_mask;
    mantissa = x & NumericUtils<float>::mant_mask;
    exponent = (head >> NumericUtils<float>::mant) & NumericUtils<float>::exp_mask;
    sign     = head >> (NumericUtils<float>::mant + NumericUtils<float>::exp);
    bias     = NumericUtils<float>::bias;

    if(x == 0)
    {
        return 0b0;
    }

    const int mini_bias                  = NumericUtils<T>::bias;
    const int mini_denormal_act_exponent = 1 - mini_bias;

    int act_exponent, out_exponent, exponent_diff;

    bool is_subnorm = false;

    if(exponent == 0)
    {
        act_exponent  = exponent - bias + 1;
        exponent_diff = mini_denormal_act_exponent - act_exponent;
        is_subnorm    = true;
    }
    else
    {
        act_exponent = exponent - bias;
        if(act_exponent <= mini_denormal_act_exponent)
        {
            exponent_diff = mini_denormal_act_exponent - act_exponent;
            is_subnorm    = true;
        }
        else
        {
            exponent_diff = 0;
        }
        mantissa += (1UL << mfmt);
    }

    auto shift_amount = (mfmt - NumericUtils<T>::mant + exponent_diff);
    shift_amount      = (shift_amount >= 64) ? 63 : shift_amount;
    bool midpoint     = (mantissa & ((1UL << shift_amount) - 1)) == (1UL << (shift_amount - 1));

    float min_subnorm = NumericLimits<T>::DataMinSubnorm() * (sign ? -1 : 1);

    if(is_subnorm && std::abs(value) < std::abs(min_subnorm))
    {
        // closer to 0
        if(std::abs(value) <= std::abs(min_subnorm - value))
            return sign << (NumericUtils<T>::exp + NumericUtils<T>::mant);
        else
            return 1 | (sign << (NumericUtils<T>::exp + NumericUtils<T>::mant));
    }

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << mfmt);
    out_exponent      = (act_exponent + exponent_diff) + mini_bias - (implicit_one ? 0 : 1);

    uint32_t drop_mask = (1UL << (mfmt - NumericUtils<T>::mant)) - 1;
    bool odd           = mantissa & (1UL << (mfmt - NumericUtils<T>::mant));
    mantissa += (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa) & drop_mask;

    if(out_exponent == 0)
    {
        if((1UL << mfmt) & mantissa)
        {
            out_exponent = 1;
        }
    }
    else
    {
        if((1UL << (mfmt + 1)) & mantissa)
        {
            mantissa >>= 1;
            out_exponent++;
        }
    }

    mantissa >>= (mfmt - NumericUtils<T>::mant);

    if(out_exponent == 0 && mantissa == 0)
    {
        return sign << (NumericUtils<T>::exp + NumericUtils<T>::mant);
    }

    mantissa &= (1UL << NumericUtils<T>::mant) - 1;
    return (sign << (NumericUtils<T>::exp + NumericUtils<T>::mant)) |
           (out_exponent << NumericUtils<T>::mant) | mantissa;
}

template <typename T>
__host__ __device__ inline T convert_to_type_sr(float value, uint32_t seed)
{
    if(std::abs(value) > NumericLimits<T>::Max())
    {
        float max_value = NumericLimits<T>::Max();

        cvt t;

        // cppcheck-suppress redundantAssignment
        t.value_float    = max_value;
        uint max_bitwise = t.value_bitwise;

        // cppcheck-suppress redundantAssignment
        t.value_float = value;
        T sign        = t.value_bitwise >> (NumericUtils<float>::exp + NumericUtils<float>::mant);
        T exp = ((max_bitwise >> NumericUtils<float>::mant) & NumericUtils<float>::exp_mask) -
                (NumericUtils<float>::bias - NumericUtils<T>::bias);

        uint32_t mant_prev = max_bitwise >> (NumericUtils<float>::mant - NumericUtils<T>::mant);
        mant_prev &= ((1UL << NumericUtils<T>::mant) - 1);
        mant_prev--;

        mant_prev <<= (NumericUtils<float>::mant - NumericUtils<T>::mant);
        uint32_t prev_bit =
            ((max_bitwise >> NumericUtils<float>::mant) << NumericUtils<float>::mant) | mant_prev;

        t.value_bitwise = prev_bit;
        float prev_val  = t.value_float;
        float diff      = max_value - prev_val;

        float actual_max = max_value + (diff / 2);

        if(std::abs(value) < actual_max)
        {
            double d_max_value  = static_cast<double>(max_value);
            double d_actual_max = static_cast<double>(actual_max);
            double d_value      = static_cast<double>(value);
            double d_is         = std::abs(d_max_value - d_actual_max);
            double d_seed       = static_cast<double>(seed);
            double d_prob = 1.0f - (std::abs(d_value - d_max_value) / d_is); // prob to round down

            double thresh = UINT_MAX * d_prob;

            if(!get_data_has_inf<T>() || d_seed <= thresh)
                // return static_cast<T>(satConvertToType(getDataMax<DTYPE>())); //round down time
                return sign == 0 ? NumericUtils<f4_t>::data_max_positive_normal_mask
                                 : NumericUtils<f4_t>::data_max_negative_normal_mask;
            else
            {
                exp++;
                return sign << ((NumericUtils<T>::exp + NumericUtils<T>::mant)) // inf
                       | (exp << NumericUtils<T>::mant);
            }
        }
        else
        {
            if(!get_data_has_inf<T>())
                return (1 << (NumericUtils<T>::mant + NumericUtils<T>::exp)) - 1;
            else
            {
                exp++;
                return sign << ((NumericUtils<T>::exp + NumericUtils<T>::mant)) // inf
                       | (exp << NumericUtils<T>::mant);
            }
        }
    }

    uint32_t f32 = bit_cast<uint32_t>(value);

    auto f32_mant = f32 & NumericUtils<float>::mant_mask;
    auto head     = f32 & NumericUtils<float>::head_mask;
    auto f32_exp  = (head >> NumericUtils<float>::mant) & NumericUtils<float>::exp_mask;

    auto sign_bit = head >> (NumericUtils<float>::mant + NumericUtils<float>::exp);
    auto sign     = sign_bit << (NumericUtils<T>::exp + NumericUtils<T>::mant);

    f32_exp      = static_cast<int32_t>(f32_exp) - NumericUtils<float>::bias;
    int32_t exp  = f32_exp;
    auto mant    = f32_mant;
    bool subnorm = false;

    if(f32 == 0)
        return 0b0;

    if(exp >= NumericUtils<T>::unbiased_exp_min)
    {
        mant = f32_mant;
    }
    // if the exponent bit is 8, then the subnormal is exactly the same as f32
    else if(exp < NumericUtils<T>::unbiased_exp_min &&
            NumericUtils<T>::exp < NumericUtils<float>::exp)
    {
        subnorm   = true;
        auto diff = static_cast<uint32_t>(NumericUtils<T>::unbiased_exp_min - exp);
        if(diff >= 32)
        {
            mant     = 0;
            f32_mant = 0;
        }
        else
        {
            f32_mant |= static_cast<uint32_t>(1) << NumericUtils<float>::mant;
            f32_mant >>= diff;
        }
        exp  = 0;
        mant = f32_mant;
    }

    uint32_t sr_shift = NumericUtils<T>::sr_shift;

    // For stochastic-rounding we add the aligned random value to the
    // mantissa and then truncate (RTZ).
    mant += seed >> sr_shift;

    // Increment exponent when mantissa overflows due to rounding
    if(mant >= static_cast<uint32_t>(1) << NumericUtils<float>::mant)
        ++exp;
    mant >>= (NumericUtils<float>::mant - NumericUtils<T>::mant);
    mant &= ((1 << NumericUtils<T>::mant) - 1);

    auto biased_exp = static_cast<uint32_t>(exp);
    if(!subnorm)
        biased_exp = static_cast<uint32_t>(exp + NumericUtils<T>::bias);
    biased_exp &= ((1 << NumericUtils<T>::exp) - 1);
    auto val = sign | biased_exp << NumericUtils<T>::mant | mant;
    return val;
}
} // namespace ck::utils
