// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "ck/utility/e8m0.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using namespace ck;

TEST(E8M0, DefaultConstructor)
{
    e8m0_bexp_t exp;
    EXPECT_EQ(exp.data, 0);
}

TEST(E8M0, InitConstructor)
{
    e8m0_bexp_t exp(0x7F);
    EXPECT_EQ(exp.data, 0x7F);
}

TEST(E8M0, FloatConstructor)
{
    e8m0_bexp_t exp(1.0f);
    EXPECT_EQ(exp.data, 0x7F);
}

TEST(E8M0, FloatConstructorNaN)
{
    e8m0_bexp_t exp(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(exp.data, 0xFF);
}

TEST(E8M0, FloatConstructorZero)
{
    e8m0_bexp_t exp(0.0f);
    EXPECT_EQ(exp.data, 0);
}

TEST(E8M0, ConversionToFloat)
{
    e8m0_bexp_t exp(0x7F);
    float value = type_convert<float>(exp);
    EXPECT_EQ(value, 1.0f);
}

TEST(E8M0, ConversionToFloatNaN)
{
    e8m0_bexp_t exp(0xFF);
    float value = type_convert<float>(exp);
    EXPECT_TRUE(std::isnan(value));
}

TEST(E8M0, MinValue)
{
    e8m0_bexp_t exp(0);
    EXPECT_TRUE(exp == ck::NumericLimits<e8m0_bexp_t>::Min());

    float value = type_convert<float>(exp);
    EXPECT_EQ(value, std::powf(2, -ck::NumericUtils<e8m0_bexp_t>::bias));
}

TEST(E8M0, MaxValue)
{
    e8m0_bexp_t exp(254);
    EXPECT_TRUE(exp == ck::NumericLimits<e8m0_bexp_t>::Max());

    float value = type_convert<float>(exp);
    EXPECT_EQ(value,
              std::powf(2,
                        ck::NumericLimits<e8m0_bexp_t>::Max().data -
                            ck::NumericUtils<e8m0_bexp_t>::bias));
}

TEST(E8M0, EqualityOperator)
{
    e8m0_bexp_t exp1(0x7F);
    e8m0_bexp_t exp2(0x7F);
    EXPECT_TRUE(exp1 == exp2);
}

TEST(E8M0, InequalityOperator)
{
    e8m0_bexp_t exp1(0x7F);
    e8m0_bexp_t exp2(0x80);
    EXPECT_FALSE(exp1 == exp2);
}

TEST(E8M0, EqualityOperatorNaN)
{
    e8m0_bexp_t exp1(0xFF);
    e8m0_bexp_t exp2(0xFF);
    EXPECT_FALSE(exp1 == exp2);
}

TEST(E8M0, GetExponentValue)
{
    e8m0_bexp_t exp(0x7F);
    int value = ck::utils::get_exponent_value(exp);
    EXPECT_EQ(value, 0x7F);
}
