// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_CODE_GEN_RTC
#pragma once

#include "ck/utility/numeric_limits.hpp"
#include "ck/utility/mxfp_utils.hpp"

namespace ck::utils {

/**
 * @brief Checks if an f6_t value is NaN based on the provided scale.
 *
 * For f6_t data, NaN cannot be represented directly. Instead, this function
 * determines NaN by checking if the scale is set to a quiet NaN.
 *
 * @param scale     The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param dataBytes The f6_t value to check (unused in this implementation).
 * @return true if the scale indicates a NaN value, false otherwise.
 */
template <>
__host__ __device__ inline bool is_nan<f6_t>(e8m0_bexp_t const scale,
                                             f6_t const dataBytes [[maybe_unused]])
{
    // no need to check for data as it does not have NaN representation
    return scale.is_nan();
}

/**
 * @brief Checks if an bf6_t value is NaN based on the provided scale.
 *
 * For bf6_t data, NaN cannot be represented directly. Instead, this function
 * determines NaN by checking if the scale is set to a quiet NaN.
 *
 * @param scale     The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param dataBytes The bf6_t value to check (unused in this implementation).
 * @return true if the scale indicates a NaN value, false otherwise.
 */
template <>
__host__ __device__ inline bool is_nan<bf6_t>(e8m0_bexp_t const scale,
                                              bf6_t const dataBytes [[maybe_unused]])
{
    // no need to check for data as it does not have NaN representation
    return scale.is_nan();
}

/**
 * @brief Checks if an f6_t value is infinite.
 *
 * Because f6_t does not support infinite values, this function always returns false.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param data  The f6_t value to check.
 * @return      Always false, as infinity is not represented in f6_t.
 */
template <>
__host__ __device__ inline bool is_inf<f6_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                             f6_t const data [[maybe_unused]])
{
    // no inf representation for fp6
    return false;
}

/**
 * @brief Checks if an bf6_t value is infinite.
 *
 * Because bf6_t does not support infinite values, this function always returns false.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param data  The bf6_t value to check.
 * @return      Always false, as infinity is not represented in bf6_t.
 */
template <>
__host__ __device__ inline bool is_inf<bf6_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                              bf6_t const data [[maybe_unused]])
{
    // no inf representation for bf6
    return false;
}

/**
 * @brief Checks whether an f6_t value is zero.
 *
 * If the specified f6_t is NaN, this function returns false.
 * Otherwise, it masks out the sign bits and checks if the remaining bits
 * are zero.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param data  The f6_t value to check.
 * @return true if the value is zero; otherwise false.
 */
template <>
__host__ __device__ inline bool is_zero<f6_t>(e8m0_bexp_t const scale, f6_t const data)
{
    if(is_nan<f6_t>(scale, data))
        return false;

    // no need to check for scale as it does not have a 0 representation
    f6_t result = (data & 0b00111111) & NumericUtils<f6_t>::set_sign_mask;

    return result == 0b0;
}

/**
 * @brief Checks whether an bf6_t value is zero.
 *
 * If the specified bf6_t is NaN, this function returns false.
 * Otherwise, it masks out the sign bits and checks if the remaining bits
 * are zero.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param data  The bf6_t value to check.
 * @return true if the value is zero; otherwise false.
 */
template <>
__host__ __device__ inline bool is_zero<bf6_t>(e8m0_bexp_t const scale, bf6_t const data)
{
    if(is_nan<bf6_t>(scale, data))
        return false;

    // no need to check for scale as it does not have a 0 representation
    bf6_t result = (data & 0b00111111) & NumericUtils<bf6_t>::set_sign_mask;

    return result == 0b0;
}

/**
 * @brief Converts an f6_t value to a float based on an e8m0_bexp_t scale factor.
 *
 * Checks if the f6_t value is NaN or zero before performing the conversion.
 * Applies the exponent from the scale to compute the final float result.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for f6_t.
 * @param data  The f6_t value to convert.
 * @return      The converted float value.
 */
template <>
__host__ __device__ inline float to_float<f6_t>(e8m0_bexp_t const scale, f6_t const data)
{
    if(is_nan<f6_t>(scale, data))
        return NumericLimits<float>::QuietNaN();

    if(is_zero<f6_t>(scale, data))
        return 0.0f;

    f6_t prepared_data = data & 0b00111111;

    int scale_exp = get_exponent_value<e8m0_bexp_t>(scale);

    return convert_to_float<f6_t>(prepared_data, scale_exp);
}

/**
 * @brief Converts an bf6_t value to a float based on an e8m0_bexp_t scale factor.
 *
 * Checks if the bf6_t value is NaN or zero before performing the conversion.
 * Applies the exponent from the scale to compute the final float result.
 *
 * @param scale The exponent scale factor (e8m0_bexp_t) used for bf6_t.
 * @param data  The bf6_t value to convert.
 * @return      The converted float value.
 */
template <>
__host__ __device__ inline float to_float<bf6_t>(e8m0_bexp_t const scale, bf6_t const data)
{
    if(is_nan<bf6_t>(scale, data))
        return NumericLimits<float>::QuietNaN();

    if(is_zero<bf6_t>(scale, data))
        return 0.0f;

    bf6_t prepared_data = data & 0b00111111;

    int scale_exp = get_exponent_value<e8m0_bexp_t>(scale);

    return convert_to_float<bf6_t>(prepared_data, scale_exp);
}

/**
 * @brief Converts a float to f6_t with saturation.
 *
 * If the input is NaN or exceeds the representable range for f6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated f6_t value.
 */
template <>
__host__ __device__ inline f6_t sat_convert_to_type<f6_t>(float value)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
    {

        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;
    }

    if(std::abs(value) > NumericLimits<f6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;

    f6_t res = convert_to_type<f6_t>(value);

    if(std::abs(to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f6_t>::DataMinSubnorm())
        return sign ? NumericUtils<f6_t>::negative_zero_mask
                    : NumericUtils<f6_t>::positive_zero_mask;

    return res;
}

/**
 * @brief Converts a float to bf6_t with saturation.
 *
 * If the input is NaN or exceeds the representable range for bf6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated bf6_t value.
 */
template <>
__host__ __device__ inline bf6_t sat_convert_to_type<bf6_t>(float value)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
    {

        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;
    }

    if(std::abs(value) > NumericLimits<bf6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;

    bf6_t res = convert_to_type<bf6_t>(value);

    if(std::abs(to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<bf6_t>::DataMinSubnorm())
        return sign ? NumericUtils<bf6_t>::negative_zero_mask
                    : NumericUtils<bf6_t>::positive_zero_mask;

    return res;
}

/**
 * @brief Converts a float to f6_t with saturation and stochastic rounding.
 *
 * If the input is NaN or exceeds the representable range for f6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated f6_t value.
 */
template <>
__host__ __device__ inline f6_t sat_convert_to_type_sr<f6_t>(float value, uint32_t seed)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;

    if(std::abs(value) > NumericLimits<f6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f6_t>::data_max_negative_normal_mask
                    : NumericUtils<f6_t>::data_max_positive_normal_mask;

    f6_t res = convert_to_type_sr<f6_t>(value, seed);

    if(std::abs(to_float<f6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f6_t>::DataMinSubnorm())
        return sign ? NumericUtils<f6_t>::negative_zero_mask
                    : NumericUtils<f6_t>::positive_zero_mask;

    return res;
}

/**
 * @brief Converts a float to f6_t with saturation and stochastic rounding.
 *
 * If the input is NaN or exceeds the representable range for f6_t, returns
 * the corresponding max normal mask. Handles subnormal cases by returning
 * zero with the appropriate sign.
 *
 * @param value The float value to be converted.
 * @return      The saturated f6_t value.
 */
template <>
__host__ __device__ inline bf6_t sat_convert_to_type_sr<bf6_t>(float value, uint32_t seed)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;
    if(std::abs(value) > NumericLimits<bf6_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<bf6_t>::data_max_negative_normal_mask
                    : NumericUtils<bf6_t>::data_max_positive_normal_mask;

    bf6_t res = convert_to_type_sr<bf6_t>(value, seed);

    if(std::abs(to_float<bf6_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<bf6_t>::DataMinSubnorm())
        return sign ? NumericUtils<bf6_t>::negative_zero_mask
                    : NumericUtils<bf6_t>::positive_zero_mask;

    return res;
}
} // namespace ck::utils
#endif
