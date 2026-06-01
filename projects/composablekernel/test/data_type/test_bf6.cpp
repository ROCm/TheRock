// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/env.hpp"
#include "ck/utility/scaled_type_convert.hpp"
#include "ck/library/utility/device_memory.hpp"

using ::ck::DeviceMem;

using ck::bf6_convert_rne;
using ck::bf6_convert_sr;
using ck::bf6_t;
using ck::bf6x16_pk_t;
using ck::bf6x32_pk_t;
using ck::e8m0_bexp_t;
using ck::Number;
using ck::scaled_type_convert;
using ck::type_convert;
using ck::vector_type;

TEST(BF6, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Min(), bf6_t(0b001000));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Max(), bf6_t(0b011111));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::Lowest(), bf6_t(0b111111));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::MinSubnorm(), bf6_t(0b000001));
    EXPECT_EQ(ck::NumericLimits<bf6_t>::MaxSubnorm(), bf6_t(0b000011));
}

TEST(BF6, ConvertFP32Nearest)
{
    // set maximum bf6 value
    float max_bf6 = 28.0f;
    // convert 0 float to bf6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(bf6_convert_rne(0.0f)), 0.0f);
    // convert max_bf6 to float and check if equal to max_bf6
    ASSERT_NEAR(max_bf6, type_convert<float>(bf6_convert_rne(max_bf6)), 0.0f);
    // convert maximal float to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(
        max_bf6, type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::max())), 0.0f);
    // convert float Inf to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::infinity())),
                0.0f);

    // convert float +/-30 to bf6 and back, check if clipped to +/-max_bf6
    ASSERT_NEAR(-max_bf6, type_convert<float>(bf6_convert_rne(-30.0f)), 0.0f);
    ASSERT_NEAR(max_bf6, type_convert<float>(bf6_convert_rne(30.0f)), 0.0f);

    // convert float value less than bf6 subnorm to bf6 and back, check if equal to 0.0
    float less_than_subnorm = 0.03125f;
    ASSERT_NEAR(0.0f, type_convert<float>(bf6_convert_rne(less_than_subnorm)), 0.0f);
    // convert float NaN to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::quiet_NaN())),
                0.0f);
    // positive norm float value to bf6 and back, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_rne(pos_float)), 0.0f);
    // negative norm float value to bf6 and back, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_rne(neg_float)), 0.0f);
    // positive subnorm float value to bf6 and back, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_rne(pos_float)), 0.0f);
    // negative subnorm float value to bf6 and back, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_rne(neg_float)), 0.0f);
}

TEST(BF6, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum bf6 value
    float max_bf6 = 28.0f;
    // convert 0 float to bf6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(bf6_convert_sr(0.0f)), abs_tol);
    // convert maximal bf6_t to float and check if equal to max_bf6
    ASSERT_NEAR(max_bf6, type_convert<float>(bf6_convert_sr(max_bf6)), abs_tol);
    // convert maximal float to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(
        max_bf6, type_convert<float>(bf6_convert_sr(std::numeric_limits<float>::max())), abs_tol);
    // convert float Inf to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::infinity())),
                0.0f);
    // convert float NaN to bf6 and back, check if clipped to max_bf6
    ASSERT_NEAR(max_bf6,
                type_convert<float>(bf6_convert_rne(std::numeric_limits<float>::quiet_NaN())),
                0.0f);
    // positive norm float value to bf6 and back, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_sr(pos_float)), abs_tol);
    // negative norm float value to bf6 and back, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_sr(neg_float)), abs_tol);
    // positive subnorm float value to bf6 and back, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float, type_convert<float>(bf6_convert_sr(pos_float)), abs_tol);
    // negative subnorm float value to bf6 and back, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float, type_convert<float>(bf6_convert_sr(neg_float)), abs_tol);
}

TEST(BF6, ScaledConvertFP32Nearest)
{
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to bf6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(0.0f)), 0.0f);
    // convert 0 float to bf6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(0.0f)), 0.0f);
    // positive norm float value to bf6 and back with various scales, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(pos_float)),
                0.0f);
    // negative norm float value to bf6 and back with various scales, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(neg_float)),
                0.0f);
    // positive subnorm float value to bf6 and back with various scales, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(pos_float)),
                0.0f);
    // negative subnorm float value to bf6 and back with various scales, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_rne(neg_float)),
                0.0f);
}

TEST(BF6, ScaledConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to bf6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(0.0f)), abs_tol);
    // convert 0 float to bf6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(0.0f)), abs_tol);
    // positive norm float value to bf6 and back with various scales, check if holds
    float pos_float = 0.25f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    // negative norm float value to bf6 and back with various scales, check if holds
    float neg_float = -0.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    // positive subnorm float value to bf6 and back with various scales, check if holds
    pos_float = 0.1875f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(pos_float)),
                abs_tol);
    // negative subnorm float value to bf6 and back with various scales, check if holds
    neg_float = -0.0625f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), bf6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), bf6_convert_sr(neg_float)),
                abs_tol);
}

TEST(BF6, TestSize)
{
    ASSERT_EQ(1, sizeof(bf6_t));
    ASSERT_EQ(16, sizeof(bf6x16_pk_t));
    ASSERT_EQ(32, sizeof(bf6x32_pk_t));
    ASSERT_EQ(16, sizeof(vector_type<bf6x16_pk_t, 1>));
    ASSERT_EQ(32, sizeof(vector_type<bf6x16_pk_t, 2>));
    ASSERT_EQ(32, sizeof(vector_type<bf6x32_pk_t, 1>));
}

TEST(BF6, TestAlignment)
{
    ASSERT_EQ(1, alignof(bf6_t));
    ASSERT_EQ(16, alignof(bf6x16_pk_t));
    ASSERT_EQ(32, alignof(bf6x32_pk_t));
    ASSERT_EQ(16, alignof(vector_type<bf6x16_pk_t, 1>));
    ASSERT_EQ(32, alignof(vector_type<bf6x16_pk_t, 2>));
    ASSERT_EQ(32, alignof(vector_type<bf6x32_pk_t, 1>));
}

// test vector of 1 bf6x16_pk_t, contains 16 bf6_t
TEST(BF6, TestAsType16x1)
{
    // test size
    const int vector_size = 1;
    const int packed_size = 16;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    test_vec_t test_vec = {bf6_t(0b000000),
                           bf6_t(0b100000),
                           bf6_t(0b000001),
                           bf6_t(0b100001),
                           bf6_t(0b000010),
                           bf6_t(0b100010),
                           bf6_t(0b000011),
                           bf6_t(0b100011),
                           bf6_t(0b000100),
                           bf6_t(0b100100),
                           bf6_t(0b000101),
                           bf6_t(0b100101),
                           bf6_t(0b000110),
                           bf6_t(0b100110),
                           bf6_t(0b001011),
                           bf6_t(0b101011)};
    // reference vector
    vector_type<bf6x16_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<bf6x16_pk_t>()(Number<0>{}).unpack(i), 0);
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<bf6x16_pk_t>()(Number<i>{}) = bf6x16_pk_t{test_vec};
    });
    // copy the vector
    vector_type<bf6x16_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<bf6x16_pk_t>()(Number<0>{}).unpack(i),
                  static_cast<bf6_t>(test_vec[static_cast<int>(i)]));
    });
}

// test vector of 2 bf6x16_pk_t, contains 32 bf6_t
TEST(BF6, TestAsType16x2)
{
    // test size
    const int vector_size = 2;
    const int packed_size = 16;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    test_vec_t test_vec[2];
    test_vec[0] = {bf6_t(0b000000),
                   bf6_t(0b100000),
                   bf6_t(0b000001),
                   bf6_t(0b100001),
                   bf6_t(0b000010),
                   bf6_t(0b100010),
                   bf6_t(0b000011),
                   bf6_t(0b100011),
                   bf6_t(0b000100),
                   bf6_t(0b100100),
                   bf6_t(0b000101),
                   bf6_t(0b100101),
                   bf6_t(0b000110),
                   bf6_t(0b100110),
                   bf6_t(0b001011),
                   bf6_t(0b101011)};
    test_vec[1] = {bf6_t(0b010000),
                   bf6_t(0b110000),
                   bf6_t(0b010001),
                   bf6_t(0b110001),
                   bf6_t(0b010010),
                   bf6_t(0b110010),
                   bf6_t(0b010011),
                   bf6_t(0b110011),
                   bf6_t(0b010100),
                   bf6_t(0b110100),
                   bf6_t(0b010101),
                   bf6_t(0b110101),
                   bf6_t(0b010110),
                   bf6_t(0b110110),
                   bf6_t(0b011011),
                   bf6_t(0b111011)};
    // reference vector
    vector_type<bf6x16_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, vector_size, 1>{}([&](auto idx_vector) {
        ck::static_for<0, packed_size, 1>{}([&](auto idx_element) {
            ASSERT_EQ(
                right_vec.template AsType<bf6x16_pk_t>()(Number<idx_vector>{}).unpack(idx_element),
                0);
        });
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<bf6x16_pk_t>()(Number<i>{}) = bf6x16_pk_t{test_vec[i]};
    });
    // copy the vector
    vector_type<bf6x16_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, vector_size, 1>{}([&](auto idx_vector) {
        ck::static_for<0, packed_size, 1>{}([&](auto idx_element) {
            ASSERT_EQ(
                left_vec.template AsType<bf6x16_pk_t>()(Number<idx_vector>{}).unpack(idx_element),
                static_cast<bf6_t>(test_vec[idx_vector][static_cast<int>(idx_element)]));
        });
    });
}

// test vector of 1 bf6x32_pk_t, contains 32 bf6_t
TEST(BF6, TestAsType32x1)
{
    // test size
    const int vector_size = 1;
    const int packed_size = 32;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(32)));
    test_vec_t test_vec = {bf6_t(0b000000), bf6_t(0b100000), bf6_t(0b000001), bf6_t(0b100001),
                           bf6_t(0b000010), bf6_t(0b100010), bf6_t(0b000011), bf6_t(0b100011),
                           bf6_t(0b000100), bf6_t(0b100100), bf6_t(0b000101), bf6_t(0b100101),
                           bf6_t(0b000110), bf6_t(0b100110), bf6_t(0b001011), bf6_t(0b101011),
                           bf6_t(0b010000), bf6_t(0b110000), bf6_t(0b010001), bf6_t(0b110001),
                           bf6_t(0b010010), bf6_t(0b110010), bf6_t(0b010011), bf6_t(0b110011),
                           bf6_t(0b010100), bf6_t(0b110100), bf6_t(0b010101), bf6_t(0b110101),
                           bf6_t(0b010110), bf6_t(0b110110), bf6_t(0b011011), bf6_t(0b111011)};
    // reference vector
    vector_type<bf6x32_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<bf6x32_pk_t>()(Number<0>{}).unpack(i), 0);
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<bf6x32_pk_t>()(Number<i>{}) = bf6x32_pk_t{test_vec};
    });
    // copy the vector
    vector_type<bf6x32_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<bf6x32_pk_t>()(Number<0>{}).unpack(i),
                  static_cast<bf6_t>(test_vec[static_cast<int>(i)]));
    });
}

TEST(BF6, TestAllValues)
{

    constexpr std::array<float, 64> e3m2ValuesOCP = {
        // clang-format off
        0.0000000000, 0.0625000000, 0.1250000000, 0.1875000000,
        0.2500000000, 0.3125000000, 0.3750000000, 0.4375000000,
        0.5000000000, 0.6250000000, 0.7500000000, 0.8750000000,
        1.0000000000, 1.2500000000, 1.5000000000, 1.7500000000,
        2.0000000000, 2.5000000000, 3.0000000000, 3.5000000000,
        4.0000000000, 5.0000000000, 6.0000000000, 7.0000000000,
        8.0000000000, 10.0000000000, 12.0000000000, 14.0000000000,
        16.0000000000, 20.0000000000, 24.0000000000, 28.0000000000,
        -0.0000000000, -0.0625000000, -0.1250000000, -0.1875000000,
        -0.2500000000, -0.3125000000, -0.3750000000, -0.4375000000,
        -0.5000000000, -0.6250000000, -0.7500000000, -0.8750000000,
        -1.0000000000, -1.2500000000, -1.5000000000, -1.7500000000,
        -2.0000000000, -2.5000000000, -3.0000000000, -3.5000000000,
        -4.0000000000, -5.0000000000, -6.0000000000, -7.0000000000,
        -8.0000000000, -10.0000000000, -12.0000000000, -14.0000000000,
        -16.0000000000, -20.0000000000, -24.0000000000, -28.0000000000
        // clang-format on
    };

    constexpr uint8_t e3m2BitsOCP[] = {
        // clang-format off
        0b000000, 0b000001, 0b000010, 0b000011,
        0b000100, 0b000101, 0b000110, 0b000111,
        0b001000, 0b001001, 0b001010, 0b001011,
        0b001100, 0b001101, 0b001110, 0b001111,
        0b010000, 0b010001, 0b010010, 0b010011,
        0b010100, 0b010101, 0b010110, 0b010111,
        0b011000, 0b011001, 0b011010, 0b011011,
        0b011100, 0b011101, 0b011110, 0b011111,
        0b100000, 0b100001, 0b100010, 0b100011,
        0b100100, 0b100101, 0b100110, 0b100111,
        0b101000, 0b101001, 0b101010, 0b101011,
        0b101100, 0b101101, 0b101110, 0b101111,
        0b110000, 0b110001, 0b110010, 0b110011,
        0b110100, 0b110101, 0b110110, 0b110111,
        0b111000, 0b111001, 0b111010, 0b111011,
        0b111100, 0b111101, 0b111110, 0b111111
        // clang-format on
    };

    const bool ck_logging = ck::EnvIsEnabled(CK_ENV(CK_LOGGING));

    if(ck_logging)
        printf("BF6 Table\n");
    ck::static_for<0, 64, 1>{}([&](auto i) {
        float fp = type_convert<float>(bf6_t(e3m2BitsOCP[i]));
        ASSERT_EQ(fp, e3m2ValuesOCP[i]);

        bf6_t bf6 = type_convert<bf6_t>(e3m2ValuesOCP[i]);
        ASSERT_EQ(bf6 & 0x3F, e3m2BitsOCP[i] & 0x3F);

        if(ck_logging)
        {
            // Print the binary representation
            printf("Bits: 0b");
            for(int j = 5; j >= 0; --j)
            {
                printf("%c", (e3m2BitsOCP[i] & (1 << j)) ? '1' : '0');
            }
            printf(", 0x%02X, Value: %f\n", e3m2BitsOCP[i], e3m2ValuesOCP[i]);
        }
    });
}

__global__ void test_bf6_convert_rne(float* p_test, uint64_t* p_completed)
{
    constexpr int N = 32;
    if(p_completed == nullptr)
    {
        return;
    }

    uint64_t& i = *p_completed;
    i           = 0;

    if(p_test == nullptr)
    {
        return;
    }

    ck::float32_t float32_in(1.0f);
    ck::float32_t float32_out{};

    auto bf6x32_vec = bf6_convert_rne(float32_in);
    float32_out     = type_convert<ck::float32_t>(bf6x32_vec);

    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = float32_out[static_cast<int>(ii)]; });
    i = N;
}

TEST(MXBF6, DeviceBF6ConvertRNE)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_bf6_convert_rne<<<1, 1>>>(static_cast<float*>(device_out.GetDeviceBuffer()),
                                   static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    EXPECT_EQ(N, completed);
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { EXPECT_EQ(out[static_cast<int>(ii)], 1.0f) << "ii: " << ii << std::endl; });

    auto bf6x32_vec_tc    = ck::type_convert<bf6x32_pk_t>(ck::float32_t(1.0f));
    auto bf6x32_vec_cnstr = bf6x32_pk_t(0x0C);

    EXPECT_EQ(bf6x32_vec_tc, bf6x32_vec_cnstr);
}
