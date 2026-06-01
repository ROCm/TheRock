// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/scaled_type_convert.hpp"
#include "ck/utility/env.hpp"

using ck::e8m0_bexp_t;
using ck::f4_convert_rne;
using ck::f4_convert_sr;
using ck::f4_t;
using ck::f4x2_pk_t;
using ck::Number;
using ck::scaled_type_convert;
using ck::type_convert;
using ck::vector_type;

TEST(FP4, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<f4_t>::Min(), f4_t{0x2});
    EXPECT_EQ(ck::NumericLimits<f4_t>::Max(), f4_t{0x7});
    EXPECT_EQ(ck::NumericLimits<f4_t>::Lowest(), f4_t{0xF});
    EXPECT_EQ(ck::NumericLimits<f4_t>::MinSubnorm(), f4_t{0x1});
    EXPECT_EQ(ck::NumericLimits<f4_t>::MaxSubnorm(), f4_t{0x1});
}

TEST(FP4, ConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp4 value
    float max_fp4 = 6.0f;
    // convert 0 float to fp4 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f4_convert_rne(0.0f)), abs_tol);
    // convert maximal f4_t to float and check if equal to 6.0
    ASSERT_NEAR(max_fp4, type_convert<float>(f4_convert_rne(max_fp4)), abs_tol);
    // convert maximal float to fp4 and back, check if clipped to 6.0
    ASSERT_NEAR(
        max_fp4, type_convert<float>(f4_convert_rne(std::numeric_limits<float>::max())), abs_tol);

    // convert +/-7.0 to fp4 and back, check if clipped to +/-6.0
    ASSERT_NEAR(-max_fp4, type_convert<float>(f4_convert_rne(-7.0f)), 0.0);
    ASSERT_NEAR(max_fp4, type_convert<float>(f4_convert_rne(7.0f)), 0.0);

    // positive norm float value to fp4 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_rne(pos_float)), abs_tol);
    // negative norm float value to fp4 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_rne(neg_float)), abs_tol);
    // positive subnorm float value to fp4 and back, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_rne(pos_float)), abs_tol);
    // negative subnorm float value to fp4 and back, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_rne(neg_float)), abs_tol);
}

TEST(FP4, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp4 value
    float max_fp4 = 6.0f;
    // convert 0 float to fp4 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f4_convert_sr(0.0f)), abs_tol);
    // convert maximal f4_t to float and check if equal to 6.0
    ASSERT_NEAR(max_fp4, type_convert<float>(f4_convert_sr(max_fp4)), abs_tol);
    // convert maximal float to fp4 and back, check if clipped to 6.0
    ASSERT_NEAR(
        max_fp4, type_convert<float>(f4_convert_sr(std::numeric_limits<float>::max())), abs_tol);
    // positive norm float value to fp4 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_sr(pos_float)), abs_tol);
    // negative norm float value to fp4 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_sr(neg_float)), abs_tol);
    // positive subnorm float value to fp4 and back, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float, type_convert<float>(f4_convert_sr(pos_float)), abs_tol);
    // negative subnorm float value to fp4 and back, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f4_convert_sr(neg_float)), abs_tol);
}

TEST(FP4, ScaledConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp4 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_rne(0.0f)), abs_tol);
    // convert 0 float to fp4 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_rne(0.0f)), abs_tol);
    // positive norm float value to fp4 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_rne(pos_float)),
                abs_tol);
    // negative norm float value to fp4 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_rne(neg_float)),
                abs_tol);
    // positive subnorm float value to fp4 and back with various scales, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_rne(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_rne(pos_float)),
                abs_tol);
    // negative subnorm float value to fp4 and back with various scales, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_rne(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_rne(neg_float)),
                abs_tol);
}

TEST(FP4, ScaledConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp4 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_sr(0.0f)), abs_tol);
    // convert 0 float to fp4 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_sr(0.0f)), abs_tol);
    // positive norm float value to fp4 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_sr(pos_float)),
                abs_tol);
    // negative norm float value to fp4 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_sr(neg_float)),
                abs_tol);
    // positive subnorm float value to fp4 and back with various scales, check if holds
    pos_float = 0.5f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_sr(pos_float)),
                abs_tol);
    // negative subnorm float value to fp4 and back with various scales, check if holds
    neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f4_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f4_convert_sr(neg_float)),
                abs_tol);
}

TEST(FP4, TestSize)
{
    ASSERT_EQ(1, sizeof(f4x2_pk_t));
    ASSERT_EQ(1, sizeof(vector_type<f4x2_pk_t, 1>));
    ASSERT_EQ(2, sizeof(vector_type<f4x2_pk_t, 2>));
    ASSERT_EQ(4, sizeof(vector_type<f4x2_pk_t, 4>));
    ASSERT_EQ(8, sizeof(vector_type<f4x2_pk_t, 8>));
    ASSERT_EQ(16, sizeof(vector_type<f4x2_pk_t, 16>));
    ASSERT_EQ(32, sizeof(vector_type<f4x2_pk_t, 32>));
}

TEST(FP4, TestAlignment)
{
    ASSERT_EQ(1, alignof(f4x2_pk_t));
    ASSERT_EQ(1, alignof(vector_type<f4x2_pk_t, 1>));
    ASSERT_EQ(2, alignof(vector_type<f4x2_pk_t, 2>));
    ASSERT_EQ(4, alignof(vector_type<f4x2_pk_t, 4>));
    ASSERT_EQ(8, alignof(vector_type<f4x2_pk_t, 8>));
    ASSERT_EQ(16, alignof(vector_type<f4x2_pk_t, 16>));
    ASSERT_EQ(32, alignof(vector_type<f4x2_pk_t, 32>));
}

// test vector of 1 f4x2_pk_t, contains 2 f4_t
TEST(FP4, TestAsType1)
{
    // test size
    const int size                        = 1;
    std::vector<f4x2_pk_t::type> test_vec = {f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001}};
    // reference vector
    vector_type<f4x2_pk_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}), 0);
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f4x2_pk_t>()(Number<i>{}) =
            f4x2_pk_t{}.pack(test_vec.at(i), test_vec.at(i + 1));
    });
    // copy the vector
    vector_type<f4x2_pk_t, size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}),
                  test_vec.at(i));
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}),
                  test_vec.at(i + 1));
    });
}

// test vector of 2 f4x2_pk_t, contains 4 f4_t
TEST(FP4, TestAsType2)
{
    // test size
    const int size                        = 2;
    std::vector<f4x2_pk_t::type> test_vec = {f4x2_pk_t::type{0b0010},
                                             f4x2_pk_t::type{0b1001},
                                             f4x2_pk_t::type{0b0001},
                                             f4x2_pk_t::type{0b0111}};
    // reference vector
    vector_type<f4x2_pk_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}), 0);
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f4x2_pk_t>()(Number<i>{}) =
            f4x2_pk_t{}.pack(test_vec.at(i), test_vec.at(i + 1));
    });
    // copy the vector
    vector_type<f4x2_pk_t, size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}),
                  test_vec.at(i));
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}),
                  test_vec.at(i + 1));
    });
}

// test vector of 4 f4x2_pk_t, contains 8 f4_t
TEST(FP4, TestAsType4)
{
    // test size
    const int size                        = 4;
    std::vector<f4x2_pk_t::type> test_vec = {f4x2_pk_t::type{0b0010},
                                             f4x2_pk_t::type{0b1001},
                                             f4x2_pk_t::type{0b0001},
                                             f4x2_pk_t::type{0b0111},
                                             f4x2_pk_t::type{0b1010},
                                             f4x2_pk_t::type{0b0001},
                                             f4x2_pk_t::type{0b1001},
                                             f4x2_pk_t::type{0b1111}};
    // reference vector
    vector_type<f4x2_pk_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}), 0);
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f4x2_pk_t>()(Number<i>{}) =
            f4x2_pk_t{}.pack(test_vec.at(i), test_vec.at(i + 1));
    });
    // copy the vector
    vector_type<f4x2_pk_t, size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}),
                  test_vec.at(i));
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}),
                  test_vec.at(i + 1));
    });
}

// test vector of 8 f4x2_pk_t, contains 16 f4_t
TEST(FP4, TestAsType8)
{
    // test size
    const int size                        = 8;
    std::vector<f4x2_pk_t::type> test_vec = {f4x2_pk_t::type{0b0010},
                                             f4x2_pk_t::type{0b1001},
                                             f4x2_pk_t::type{0b0001},
                                             f4x2_pk_t::type{0b0111},
                                             f4x2_pk_t::type{0b1010},
                                             f4x2_pk_t::type{0b0001},
                                             f4x2_pk_t::type{0b1001},
                                             f4x2_pk_t::type{0b1111},
                                             f4x2_pk_t::type{0b0001},
                                             f4x2_pk_t::type{0b0111},
                                             f4x2_pk_t::type{0b1010},
                                             f4x2_pk_t::type{0b0001},
                                             f4x2_pk_t::type{0b0010},
                                             f4x2_pk_t::type{0b1001},
                                             f4x2_pk_t::type{0b1001},
                                             f4x2_pk_t::type{0b1111}};
    // reference vector
    vector_type<f4x2_pk_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}), 0);
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f4x2_pk_t>()(Number<i>{}) =
            f4x2_pk_t{}.pack(test_vec.at(i), test_vec.at(i + 1));
    });
    // copy the vector
    vector_type<f4x2_pk_t, size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}),
                  test_vec.at(i));
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}),
                  test_vec.at(i + 1));
    });
}

// test vector of 16 f4x2_pk_t, contains 32 f4_t
TEST(FP4, TestAsType16)
{
    // test size
    const int size                        = 16;
    std::vector<f4x2_pk_t::type> test_vec = {
        f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b1111}, f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111}};
    // reference vector
    vector_type<f4x2_pk_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}), 0);
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f4x2_pk_t>()(Number<i>{}) =
            f4x2_pk_t{}.pack(test_vec.at(i), test_vec.at(i + 1));
    });
    // copy the vector
    vector_type<f4x2_pk_t, size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}),
                  test_vec.at(i));
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}),
                  test_vec.at(i + 1));
    });
}

// test vector of 32 f4x2_pk_t, contains 64 f4_t
TEST(FP4, TestAsType32)
{
    // test size
    const int size                        = 32;
    std::vector<f4x2_pk_t::type> test_vec = {
        f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b1111}, f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010},
        f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111}, f4x2_pk_t::type{0b0010},
        f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0111},
        f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b1111}, f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0111},
        f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001}, f4x2_pk_t::type{0b0010},
        f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111},
        f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1111}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0111}, f4x2_pk_t::type{0b1010}, f4x2_pk_t::type{0b0001},
        f4x2_pk_t::type{0b0010}, f4x2_pk_t::type{0b1001}, f4x2_pk_t::type{0b1001},
        f4x2_pk_t::type{0b1111}};
    // reference vector
    vector_type<f4x2_pk_t, size> right_vec;
    // check default CTOR
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}), 0);
        ASSERT_EQ(
            right_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}), 0);
    });
    // assign test values to the vector
    ck::static_for<0, size, 1>{}([&](auto i) {
        right_vec.template AsType<f4x2_pk_t>()(Number<i>{}) =
            f4x2_pk_t{}.pack(test_vec.at(i), test_vec.at(i + 1));
    });
    // copy the vector
    vector_type<f4x2_pk_t, size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<0>{}),
                  test_vec.at(i));
        ASSERT_EQ(left_vec.template AsType<f4x2_pk_t>()(Number<i>{}).template unpack<>(Number<1>{}),
                  test_vec.at(i + 1));
    });
}

TEST(FP4, TestAllValues)
{
    constexpr std::array<float, 16> e2m1ValuesOCP = {
        // clang-format off
        0.0000000000, 0.5000000000,
        1.0000000000, 1.5000000000,
        2.0000000000, 3.0000000000,
        4.0000000000, 6.0000000000,
        -0.0000000000, -0.5000000000,
        -1.0000000000, -1.5000000000,
        -2.0000000000, -3.0000000000,
        -4.0000000000, -6.0000000000
        // clang-format on
    };

    constexpr uint8_t e2m1BitsOCP[] = {
        // clang-format off
        0b0000, 0b0001,
        0b0010, 0b0011,
        0b0100, 0b0101,
        0b0110, 0b0111,
        0b1000, 0b1001,
        0b1010, 0b1011,
        0b1100, 0b1101,
        0b1110, 0b1111
        // clang-format on
    };

    const bool ck_logging = ck::EnvIsEnabled(CK_ENV(CK_LOGGING));

    if(ck_logging)
        printf("FP4 Table\n");
    ck::static_for<0, 16, 1>{}([&](auto i) {
        float fp = type_convert<float>(f4_t(e2m1BitsOCP[i]));
        ASSERT_EQ(fp, e2m1ValuesOCP[i]);

        f4_t fp4 = type_convert<f4_t>(e2m1ValuesOCP[i]);
        ASSERT_EQ(fp4 & 0xF, e2m1BitsOCP[i] & 0xF);
        if(ck_logging)
        {
            // Print the binary representation
            printf("Bits: 0b");
            for(int j = 3; j >= 0; --j)
            {
                printf("%c", (e2m1BitsOCP[i] & (1 << j)) ? '1' : '0');
            }
            printf(", 0x%02X, Value: %f\n", e2m1BitsOCP[i], e2m1ValuesOCP[i]);
        }
    });
}
