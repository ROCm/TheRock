// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/numeric_utils.hpp"

namespace ck {

// fp8 rounding modes
// use standard for rounding to nearest, the faster one
// use stochastic for stochastic rounding, helps to avoid error accumulation
enum class f8_rounding_mode
{
    standard,
    stochastic
};

__host__ inline int clz(uint32_t x) { return __builtin_clz(x); }
__device__ inline int clz(uint32_t x) { return __clz(x); }

} // namespace ck

namespace ck::utils {

namespace {

template <typename X, typename Y, bool negative_zero_nan, bool clip, bool stoch>
__host__ __device__ Y run_cast_to_f8(X x, uint32_t rng)
{
    // fp8/bf8 exponent/mantissa layout
    constexpr int out_exp  = NumericUtils<Y>::exp;
    constexpr int out_mant = NumericUtils<Y>::mant;

    // original type exponent/mantissa layout
    constexpr int in_exp  = NumericUtils<X>::exp;
    constexpr int in_mant = NumericUtils<X>::mant;

    int exponent, bias;
    uint32_t head, mantissa, sign;
    // nan code is same for float and half
    constexpr uint8_t nan_code  = 0x80;
    constexpr uint32_t nan_mask = NumericUtils<X>::nan_mask;

    // convert to bitwise
    using T_bitwise     = typename NumericUtils<X>::bitwise_type;
    T_bitwise x_bitwise = bit_cast<T_bitwise>(x);

    // unpack the input, depends on datatype
    head     = x_bitwise & NumericUtils<X>::head_mask;
    mantissa = x_bitwise & NumericUtils<X>::mant_mask;
    exponent = (head >> in_mant) & NumericUtils<X>::exp_mask;
    sign     = head >> (in_exp + in_mant);
    bias     = NumericUtils<X>::bias;

    uint32_t signed_inf   = (sign << (in_exp + in_mant)) + (((1 << in_exp) - 1) << in_mant);
    uint32_t drop_mask    = (1 << (in_mant - out_mant)) - 1;
    constexpr int max_exp = (1 << out_exp) - (negative_zero_nan ? 1 : 2);

    if constexpr(negative_zero_nan)
    {
        if((x_bitwise & nan_mask) == nan_mask)
            return Y{nan_code};
    }
    else
    {
        if((x_bitwise & nan_mask) == nan_mask)
            return Y{static_cast<uint8_t>(signed_inf + (mantissa != 0 ? 1 : 0))};
    }

    // check if x is 0.0
    if(x_bitwise == 0)
        return Y{0};

    // First need to check if it is normal or denorm as there is a difference of implict 1
    // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
    // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
    // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
    // exponent and mantissa again3

    // For IEEE bias mode, the bias is 2^(k-1)-1 where k is the width of exponent bits
    const int out_bias                  = (1 << (out_exp - 1)) - 1 + (negative_zero_nan ? 1 : 0);
    const int out_denormal_act_exponent = 1 - out_bias; // actual exponent of f8 denormal
    // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
    // out_exponent is the converted f8 exponent with bias encoding
    // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
    // the difference needs to be adjusted and mantissa shifted
    int act_exponent, out_exponent, exponent_diff;

    if(exponent == 0)
    { // fp32/fp16 is in denormal.
        /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16
here. In this case, f8 is usually in denormal. But there could be exceptions. fp16 denormal has
exponent bias 15 while bf8 with NANOO has exponent bias 16. It means that there are some numbers in
fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal.
In this case, the fp16 mantissa should be shift left by 1 */
        act_exponent  = exponent - bias + 1;
        exponent_diff = out_denormal_act_exponent -
                        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
    }
    else
    { // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
        if(act_exponent <= out_denormal_act_exponent)
        {
            /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
   For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
   actual exponent is -7, it is actually larger due to the implict 1,
   Therefore it needs to be adjust to -6 and mantissa shift right by 1.
   So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
            exponent_diff = out_denormal_act_exponent - act_exponent;
        }
        else
        { // both fp32/fp16 and f8 are in normal range
            exponent_diff =
                0; // exponent_diff=0 does not mean there is no difference for this case,
            // act_exponent could be larger. Just that it does not need shift mantissa
        }
        mantissa += (1 << in_mant); // Add the implicit 1 into mantissa
    }

    bool midpoint = (mantissa & ((1 << (in_mant - out_mant + exponent_diff)) - 1)) ==
                    (1 << (in_mant - out_mant + exponent_diff - 1));
    /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we
 shift right as shift right could rip off some residual part and make something not midpoint look
 like midpoint. For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than
 midpoint, but after shift right by 4 bits, it would look like midpoint. */

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << in_mant);
    // if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
    out_exponent =
        (act_exponent + exponent_diff) /*actual f8 exponent*/ + out_bias - (implicit_one ? 0 : 1);

    // Now we have the exponent and mantissa adjusted
    bool odd =
        mantissa &
        (1 << (in_mant - out_mant)); // if the least significant bit that is not truncated is 1
    mantissa += (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa)) & drop_mask;

    // Now we deal with overflow
    if(out_exponent == 0)
    {
        if((1 << in_mant) & mantissa)
        {
            out_exponent = 1; // denormal overflow to become normal, promote exponent
            // No need to make 1 implicit now as it will be addressed later
        }
    }
    else
    {
        if((1 << (in_mant + 1)) & mantissa)
        {
            mantissa >>= 1;
            out_exponent++;
            // No need to make 1 implicit now as it will be addressed later
        }
    }

    mantissa >>= (in_mant - out_mant);

    if(out_exponent > max_exp)
    {
        if constexpr(clip)
        {
            mantissa     = (1 << out_mant) - 1;
            out_exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }

    // check if x is 0.0 or -0.0
    if(out_exponent == 0 && mantissa == 0)
        return Y{negative_zero_nan ? 0 : static_cast<uint8_t>(sign << (out_exp + out_mant))};
    mantissa &= (1 << out_mant) - 1;
    return Y{static_cast<uint8_t>((sign << (out_exp + out_mant)) | (out_exponent << out_mant) |
                                  mantissa)};
}

template <typename X, typename Y, bool negative_zero_nan>
__host__ __device__ Y run_cast_from_f8(X x)
{
    // fp8/bf8 exponent/mantissa layout
    constexpr int in_exp  = NumericUtils<X>::exp;
    constexpr int in_mant = NumericUtils<X>::mant;

    // resulting type exponent/mantissa layout
    constexpr int out_exp  = NumericUtils<Y>::exp;
    constexpr int out_mant = NumericUtils<Y>::mant;

    // prepare the codes
    constexpr uint8_t nan_code = 0x80;
    using T_bitwise            = typename NumericUtils<Y>::bitwise_type;

    constexpr T_bitwise Inf_bitwise    = NumericUtils<Y>::Inf;
    constexpr T_bitwise NegInf_bitwise = NumericUtils<Y>::NegInf;
    constexpr T_bitwise NaN_bitwise    = NumericUtils<Y>::NaN;
    constexpr T_bitwise Neg0_bitwise   = NumericUtils<Y>::Neg0;

    constexpr Y Inf    = bit_cast<Y>(Inf_bitwise);
    constexpr Y NegInf = bit_cast<Y>(NegInf_bitwise);
    constexpr Y NaN    = bit_cast<Y>(NaN_bitwise);
    constexpr Y Neg0   = bit_cast<Y>(Neg0_bitwise);

    // check if x is 0.0
    if(!static_cast<uint8_t>(x))
        return static_cast<Y>(0);

    // unpack the input
    uint32_t sign     = static_cast<uint8_t>(x) >> (in_exp + in_mant);
    uint32_t mantissa = static_cast<uint8_t>(x) & ((1 << in_mant) - 1);
    int exponent      = (static_cast<uint8_t>(x) & 0x7F) >> in_mant;

    constexpr int exp_low_cutoff =
        (1 << (out_exp - 1)) - (1 << (in_exp - 1)) + 1 - (negative_zero_nan ? 1 : 0);
    T_bitwise retval;

    if constexpr(negative_zero_nan)
    {
        if(static_cast<uint8_t>(x) == nan_code)
            return NaN;
    }
    else
    {
        if(static_cast<uint8_t>(x) == nan_code)
            return Neg0;
        if(exponent == ((1 << in_exp) - 1))
            return (mantissa == 0) ? (sign ? NegInf : Inf) : NaN;
    }

    if constexpr((NumericUtils<Y>::mant == 10) && (NumericUtils<X>::mant == 2) &&
                 !negative_zero_nan)
    {
        retval = x;
        retval <<= 8;
        return bit_cast<Y>(retval);
    }

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + clz(mantissa) - (32 - in_mant);
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= ((1 << in_mant) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= out_mant - in_mant;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << out_mant;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval = (sign << (out_exp + out_mant)) | (exponent << out_mant) | mantissa;
    return bit_cast<Y>(retval);
}

} // namespace

template <typename X, typename Y, bool negative_zero_nan, bool clip, bool stoch>
__host__ __device__ Y cast_to_f8(X x, uint32_t rng)
{
    // check datatypes
    constexpr bool is_half  = is_same<X, half_t>::value;
    constexpr bool is_float = is_same<X, float>::value;
    static_assert(is_half || is_float, "Only half and float can be casted.");

    return run_cast_to_f8<X, Y, negative_zero_nan, clip, stoch>(x, rng);
}

template <typename X, typename Y, bool negative_zero_nan>
__host__ __device__ Y cast_from_f8(X x)
{
    // check datatype
    constexpr bool is_half  = is_same<Y, half_t>::value;
    constexpr bool is_float = is_same<Y, float>::value;
    static_assert(is_half || is_float, "only half and float are supported.");

    return run_cast_from_f8<X, Y, negative_zero_nan>(x);
}

} // namespace ck::utils
