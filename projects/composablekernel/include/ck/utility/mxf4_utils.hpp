// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_CODE_GEN_RTC
#pragma once

#include "ck/utility/numeric_limits.hpp"
#include "ck/utility/mxfp_utils.hpp"

namespace ck::utils {

template <>
__host__ __device__ inline bool is_nan<f4_t>(e8m0_bexp_t const scale,
                                             f4_t const dataBytes [[maybe_unused]])
{
    // no need to check for data as it does not have NaN representation
    return scale.is_nan();
}

// no infinity representation in ocp_e2m1_mxfp4 will always return false
template <>
__host__ __device__ inline bool is_inf<f4_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                             f4_t const data [[maybe_unused]])
{
    // no inf representation for ocp_e2m1_mxfp4
    return false;
}

template <>
__host__ __device__ inline bool is_zero<f4_t>(e8m0_bexp_t const scale [[maybe_unused]],
                                              f4_t const data)
{
    // no need to check for scale as it does not have a 0 representation
    f4_t result = (data & 0b00001111) & NumericUtils<f4_t>::set_sign_mask;

    return result == 0b0;
}

template <>
__host__ __device__ inline float to_float<f4_t>(e8m0_bexp_t const scale, f4_t const data)
{
    if(is_nan<f4_t>(scale, data))
        return NumericLimits<float>::QuietNaN();

    if(is_zero<f4_t>(scale, data))
        return 0.0f;

    f4_t prepared_data = data & 0b00001111;

    int scale_exp = get_exponent_value<e8m0_bexp_t>(scale);

    return convert_to_float<f4_t>(prepared_data, scale_exp);
}

template <>
__host__ __device__ inline f4_t sat_convert_to_type<f4_t>(float value)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
    {

        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;
    }

    if(std::abs(value) > NumericLimits<f4_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    f4_t res = convert_to_type<f4_t>(value);

    if(std::abs(to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f4_t>::DataMinSubnorm())
        return sign ? NumericUtils<f4_t>::negative_zero_mask
                    : NumericUtils<f4_t>::positive_zero_mask;

    return res;
}

template <>
__host__ __device__ inline f4_t sat_convert_to_type_sr<f4_t>(float value, uint32_t seed)
{
    cvt t;
    t.value_float = value;
    uint32_t sign = t.value_bitwise >> 31;

    if(std::isnan(value))
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    if(std::abs(value) > NumericLimits<f4_t>::DataMaxNorm()) // covers inf case as well
        return sign ? NumericUtils<f4_t>::data_max_negative_normal_mask
                    : NumericUtils<f4_t>::data_max_positive_normal_mask;

    f4_t res = convert_to_type_sr<f4_t>(value, seed);

    if(std::abs(to_float<f4_t>(NumericLimits<e8m0_bexp_t>::Binary_1(), res)) <
       NumericLimits<f4_t>::DataMinSubnorm())
        return sign ? NumericUtils<f4_t>::negative_zero_mask
                    : NumericUtils<f4_t>::positive_zero_mask;

    return res;
}
} // namespace ck::utils
#endif
