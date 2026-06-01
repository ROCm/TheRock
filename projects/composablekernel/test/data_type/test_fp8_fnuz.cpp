// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ck::f8_convert_rne;
using ck::f8_convert_sr;
using ck::f8_fnuz_t;
using ck::half_t;
using ck::type_convert;

TEST(FP8FNUZ, NumericLimits)
{
    // constants given for negative zero nan mode
    EXPECT_EQ(ck::NumericLimits<f8_fnuz_t>::Min(), type_convert<f8_fnuz_t>(0x08));
    EXPECT_EQ(ck::NumericLimits<f8_fnuz_t>::Max(), type_convert<f8_fnuz_t>(0x7F));
    EXPECT_EQ(ck::NumericLimits<f8_fnuz_t>::Lowest(), type_convert<f8_fnuz_t>(0xFF));
    EXPECT_EQ(ck::NumericLimits<f8_fnuz_t>::QuietNaN(), type_convert<f8_fnuz_t>(0x80));
}

TEST(FP8FNUZ, ConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to fp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_rne<f8_fnuz_t>(0.0f)), abs_tol);
    // don't run the next test on gfx11 devices
#ifndef CK_SKIP_FLAKY_F8_TEST
    // convert minimal float to fp8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_rne<f8_fnuz_t>(std::numeric_limits<float>::min())),
                abs_tol);
#endif

    const auto max_f8_t_float = type_convert<float>(ck::NumericLimits<f8_fnuz_t>::Max());
    // convert maximal f8_fnuz_t to float and check if equal to fp8 max
    ASSERT_NEAR(
        max_f8_t_float, type_convert<float>(f8_convert_rne<f8_fnuz_t>(max_f8_t_float)), abs_tol);

    // XXX: FNUZ f8_convert_rne behavior is inconsistent.
    // Clipping large values to fp8 max (saturation to finite) contradicts converting inf float to
    // fp8 qNAN (no saturation).

    // convert maximal float to fp8 and back, check if clipped to fp8 max
    ASSERT_NEAR(max_f8_t_float,
                type_convert<float>(f8_convert_rne<f8_fnuz_t>(std::numeric_limits<float>::max())),
                abs_tol);
    // convert inf float to f8_fnuz_t and check if it is qNan
    ASSERT_EQ(ck::NumericLimits<f8_fnuz_t>::QuietNaN(),
              f8_convert_rne<f8_fnuz_t>(std::numeric_limits<float>::infinity()));
    // positive norm float value to fp8 and back, check if holds
    float pos_float = 0.017578125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_rne<f8_fnuz_t>(pos_float)), abs_tol);
    // negative norm float value to fp8 and back, check if holds
    float neg_float = -0.015625f;
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_rne<f8_fnuz_t>(neg_float)), abs_tol);
    // positive subnorm float value to fp8 and back, check if holds
    pos_float = 0.00390625f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_rne<f8_fnuz_t>(pos_float)), abs_tol);
    // negative subnorm float value to fp8 and back, check if holds
    neg_float = -0.001953125f;
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_rne<f8_fnuz_t>(neg_float)), abs_tol);
}

TEST(FP8FNUZ, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // convert 0 float to fp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_sr<f8_fnuz_t>(0.0f)), abs_tol);
    // convert minimal float to fp8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_sr<f8_fnuz_t>(std::numeric_limits<float>::min())),
                abs_tol);

    const auto max_f8_t_float = type_convert<float>(ck::NumericLimits<f8_fnuz_t>::Max());
    // convert maximal f8_fnuz_t to float and check if equal to fp8 max
    ASSERT_NEAR(
        max_f8_t_float, type_convert<float>(f8_convert_sr<f8_fnuz_t>(max_f8_t_float)), abs_tol);
    // convert maximal float to fp8 and back, check if clipped to fp8 max
    ASSERT_NEAR(max_f8_t_float,
                type_convert<float>(f8_convert_sr<f8_fnuz_t>(std::numeric_limits<float>::max())),
                abs_tol);
    // convert inf float to f8_fnuz_t and check if it is qNan
    ASSERT_EQ(ck::NumericLimits<f8_fnuz_t>::QuietNaN(),
              f8_convert_sr<f8_fnuz_t>(std::numeric_limits<float>::infinity()));
    // positive norm float value to fp8 and back, check if holds
    float pos_float = 0.017578125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_sr<f8_fnuz_t>(pos_float)), abs_tol);
    // negative norm float value to fp8 and back, check if holds
    float neg_float = -0.015625f;
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_sr<f8_fnuz_t>(neg_float)), abs_tol);
    // positive subnorm float value to fp8 and back, check if holds
    pos_float = 0.00390625f;
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_sr<f8_fnuz_t>(pos_float)), abs_tol);
    // negative subnorm float value to fp8 and back, check if holds
    neg_float = -0.001953125f;
    ASSERT_NEAR(neg_float, type_convert<float>(f8_convert_sr<f8_fnuz_t>(neg_float)), abs_tol);
}

TEST(FP8FNUZ, ConvertFP16Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-3;
    // convert 0 fp16 to fp8 and back, check if holds
    ASSERT_NEAR(half_t{0.0}, type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(half_t{0.0})), abs_tol);
    // convert minimal fp16 to fp8 and back, check if holds
    ASSERT_NEAR(ck::NumericLimits<half_t>::Min(),
                type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(ck::NumericLimits<half_t>::Min())),
                abs_tol);

    const auto max_f8_t_half = type_convert<half_t>(ck::NumericLimits<f8_fnuz_t>::Max());
    // convert maximal f8_fnuz_t to fp16 and check if equal to fp8 max
    ASSERT_NEAR(
        max_f8_t_half, type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(max_f8_t_half)), abs_tol);
    // convert maximal fp16 to fp8 and back, check if clipped to fp8 max
    ASSERT_NEAR(max_f8_t_half,
                type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(ck::NumericLimits<half_t>::Max())),
                abs_tol);
    // convert QuietNaN fp16 to f8_fnuz_t and check if it is QuietNaN
    ASSERT_EQ(ck::NumericLimits<f8_fnuz_t>::QuietNaN(),
              f8_convert_rne<f8_fnuz_t>(ck::NumericLimits<half_t>::QuietNaN()));
    // positive norm fp16 value to fp8 and back, check if holds
    half_t pos_half = half_t{0.017578125};
    ASSERT_NEAR(pos_half, type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(pos_half)), abs_tol);
    // negative norm fp16 value to fp8 and back, check if holds
    half_t neg_half = half_t{-0.015625};
    ASSERT_NEAR(neg_half, type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(neg_half)), abs_tol);
    // positive subnorm fp16 value to fp8 and back, check if holds
    pos_half = half_t{0.00390625};
    ASSERT_NEAR(pos_half, type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(pos_half)), abs_tol);
    // negative subnorm fp16 value to fp8 and back, check if holds
    neg_half = half_t{-0.001953125};
    ASSERT_NEAR(neg_half, type_convert<half_t>(f8_convert_rne<f8_fnuz_t>(neg_half)), abs_tol);
}

TEST(FP8FNUZ, ConvertFP16Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-3;
    // convert 0 fp16 to fp8 and back, check if holds
    ASSERT_NEAR(half_t{0.0}, type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(half_t{0.0})), abs_tol);
    // convert minimal fp16 to fp8 and back, check if holds
    ASSERT_NEAR(ck::NumericLimits<half_t>::Min(),
                type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(ck::NumericLimits<half_t>::Min())),
                abs_tol);

    const auto max_f8_t_half = type_convert<half_t>(ck::NumericLimits<f8_fnuz_t>::Max());
    // convert maximal f8_fnuz_t to fp16 and check if equal to fp8 max
    ASSERT_NEAR(
        max_f8_t_half, type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(max_f8_t_half)), abs_tol);
    // convert maximal fp16 to fp8 and back, check if clipped to fp8 max
    ASSERT_NEAR(max_f8_t_half,
                type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(ck::NumericLimits<half_t>::Max())),
                abs_tol);
    // convert QuietNaN fp16 to f8_fnuz_t and check if it is QuietNaN
    ASSERT_EQ(ck::NumericLimits<f8_fnuz_t>::QuietNaN(),
              f8_convert_sr<f8_fnuz_t>(ck::NumericLimits<half_t>::QuietNaN()));
    // positive norm fp16 value to fp8 and back, check if holds
    half_t pos_half = half_t{0.017578125};
    ASSERT_NEAR(pos_half, type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(pos_half)), abs_tol);
    // negative norm fp16 value to fp8 and back, check if holds
    half_t neg_half = half_t{-0.015625};
    ASSERT_NEAR(neg_half, type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(neg_half)), abs_tol);
    // positive subnorm fp16 value to fp8 and back, check if holds
    pos_half = half_t{0.00390625};
    ASSERT_NEAR(pos_half, type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(pos_half)), abs_tol);
    // negative subnorm fp16 value to fp8 and back, check if holds
    neg_half = half_t{-0.001953125};
    ASSERT_NEAR(neg_half, type_convert<half_t>(f8_convert_sr<f8_fnuz_t>(neg_half)), abs_tol);
}
