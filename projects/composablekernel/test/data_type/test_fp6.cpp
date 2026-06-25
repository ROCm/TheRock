// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/env.hpp"
#include "ck/utility/scaled_type_convert.hpp"
#include "ck/library/utility/device_memory.hpp"

using ::ck::DeviceMem;

using ck::e8m0_bexp_t;
using ck::f6_convert_rne;
using ck::f6_convert_sr;
using ck::f6_t;
using ck::f6x16_pk_t;
using ck::f6x32_pk_t;
using ck::Number;
using ck::scaled_type_convert;
using ck::type_convert;
using ck::vector_type;

TEST(FP6, NumericLimits)
{
    EXPECT_EQ(ck::NumericLimits<f6_t>::Min(), f6_t(0b001000));
    EXPECT_EQ(ck::NumericLimits<f6_t>::Max(), f6_t(0b011111));
    EXPECT_EQ(ck::NumericLimits<f6_t>::Lowest(), f6_t(0b111111));
    EXPECT_EQ(ck::NumericLimits<f6_t>::MinSubnorm(), f6_t(0b000001));
    EXPECT_EQ(ck::NumericLimits<f6_t>::MaxSubnorm(), f6_t(0b000111));
}

TEST(FP6, ConvertFP32Nearest)
{
    // set maximum fp6 value
    float max_fp6 = 7.5f;
    // convert 0 float to fp6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f6_convert_rne(0.0f)), 0.0f);
    // convert maximal f6_t to float and check if equal to max_fp6
    ASSERT_NEAR(max_fp6, type_convert<float>(f6_convert_rne(max_fp6)), 0.0f);

    // convert maximal +/-8.0 to fp6 and check if equal to +/-max_fp6
    ASSERT_NEAR(-max_fp6, type_convert<float>(f6_convert_rne(-8.0f)), 0.0f);
    ASSERT_NEAR(max_fp6, type_convert<float>(f6_convert_rne(8.0f)), 0.0f);

    // convert maximal float to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(
        max_fp6, type_convert<float>(f6_convert_rne(std::numeric_limits<float>::max())), 0.0f);
    // convert float Inf to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(
        max_fp6, type_convert<float>(f6_convert_rne(std::numeric_limits<float>::infinity())), 0.0f);
    // convert float value less than fp6 subnorm to fp6 and back, check if equal to 0.0
    float less_than_subnorm = 0.0625f;
    ASSERT_NEAR(0.0f, type_convert<float>(f6_convert_rne(less_than_subnorm)), 0.0f);
    // convert float NaN to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(max_fp6,
                type_convert<float>(f6_convert_rne(std::numeric_limits<float>::quiet_NaN())),
                0.0f);
    // positive norm float value to fp6 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_rne(pos_float)), 0.0f);
    // negative norm float value to fp6 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_rne(neg_float)), 0.0f);
    // positive subnorm float value to fp6 and back, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_rne(pos_float)), 0.0f);
    // negative subnorm float value to fp6 and back, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_rne(neg_float)), 0.0f);
}

TEST(FP6, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum fp6 value
    float max_fp6 = 7.5f;
    // convert 0 float to fp6 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f6_convert_sr(0.0f)), abs_tol);
    // convert maximal f6_t to float and check if equal to max_fp6
    ASSERT_NEAR(max_fp6, type_convert<float>(f6_convert_sr(max_fp6)), abs_tol);
    // convert maximal float to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(
        max_fp6, type_convert<float>(f6_convert_sr(std::numeric_limits<float>::max())), abs_tol);
    // convert float Inf to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(max_fp6,
                type_convert<float>(f6_convert_sr(std::numeric_limits<float>::infinity())),
                abs_tol);
    // convert float NaN to fp6 and back, check if clipped to max_fp6
    ASSERT_NEAR(max_fp6,
                type_convert<float>(f6_convert_sr(std::numeric_limits<float>::quiet_NaN())),
                abs_tol);
    // positive norm float value to fp6 and back, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_sr(pos_float)), abs_tol);
    // negative norm float value to fp6 and back, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_sr(neg_float)), abs_tol);
    // positive subnorm float value to fp6 and back, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float, type_convert<float>(f6_convert_sr(pos_float)), abs_tol);
    // negative subnorm float value to fp6 and back, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float, type_convert<float>(f6_convert_sr(neg_float)), abs_tol);
}

TEST(FP6, ScaledConvertFP32Nearest)
{
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(0.0f)), 0.0f);
    // convert 0 float to fp6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(0.0f)), 0.0f);
    // positive norm float value to fp6 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(pos_float)),
                0.0f);
    // negative norm float value to fp6 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(neg_float)),
                0.0f);
    // positive subnorm float value to fp6 and back with various scales, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(pos_float)),
                0.0f);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(pos_float)),
                0.0f);
    // negative subnorm float value to fp6 and back with various scales, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_rne(neg_float)),
                0.0f);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_rne(neg_float)),
                0.0f);
}

TEST(FP6, ScaledConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;
    // set maximum scale
    float max_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Max()); // 0xFE -> float
    // set minimum scale
    float min_scale = type_convert<float>(ck::NumericLimits<e8m0_bexp_t>::Min()); // 0x00 -> float
    // set arbitrary scale to 256.0
    float test_scale = 256.0f; // 0b10000111
    // convert 0 float to fp6 and back with maximal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(0.0f)), abs_tol);
    // convert 0 float to fp6 and back with minimal scale, check if holds
    ASSERT_NEAR(
        0.0f, scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(0.0f)), abs_tol);
    // positive norm float value to fp6 and back with various scales, check if holds
    float pos_float = 1.0f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(pos_float)),
                abs_tol);
    // negative norm float value to fp6 and back with various scales, check if holds
    float neg_float = -1.5f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(neg_float)),
                abs_tol);
    // positive subnorm float value to fp6 and back with various scales, check if holds
    pos_float = 0.125f;
    ASSERT_NEAR(pos_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(pos_float)),
                abs_tol);
    ASSERT_NEAR(pos_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(pos_float)),
                abs_tol);
    // negative subnorm float value to fp6 and back with various scales, check if holds
    neg_float = -0.25f;
    ASSERT_NEAR(neg_float * test_scale,
                scaled_type_convert<float>(e8m0_bexp_t(test_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * max_scale,
                scaled_type_convert<float>(e8m0_bexp_t(max_scale), f6_convert_sr(neg_float)),
                abs_tol);
    ASSERT_NEAR(neg_float * min_scale,
                scaled_type_convert<float>(e8m0_bexp_t(min_scale), f6_convert_sr(neg_float)),
                abs_tol);
}

TEST(FP6, TestSize)
{
    ASSERT_EQ(1, sizeof(f6_t));
    ASSERT_EQ(16, sizeof(f6x16_pk_t));
    ASSERT_EQ(32, sizeof(f6x32_pk_t));
    ASSERT_EQ(16, sizeof(vector_type<f6x16_pk_t, 1>));
    ASSERT_EQ(32, sizeof(vector_type<f6x16_pk_t, 2>));
    ASSERT_EQ(32, sizeof(vector_type<f6x32_pk_t, 1>));
}

TEST(FP6, TestAlignment)
{
    ASSERT_EQ(1, alignof(f6_t));
    ASSERT_EQ(16, alignof(f6x16_pk_t));
    ASSERT_EQ(32, alignof(f6x32_pk_t));
    ASSERT_EQ(16, alignof(vector_type<f6x16_pk_t, 1>));
    ASSERT_EQ(32, alignof(vector_type<f6x16_pk_t, 2>));
    ASSERT_EQ(32, alignof(vector_type<f6x32_pk_t, 1>));
}

// test vector of 1 f6x16_pk_t, contains 16 f6_t
TEST(FP6, TestAsType16x1)
{
    // test size
    const int vector_size = 1;
    const int packed_size = 16;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    test_vec_t test_vec = {f6_t(0b000000),
                           f6_t(0b100000),
                           f6_t(0b000001),
                           f6_t(0b100001),
                           f6_t(0b000010),
                           f6_t(0b100010),
                           f6_t(0b000011),
                           f6_t(0b100011),
                           f6_t(0b000100),
                           f6_t(0b100100),
                           f6_t(0b000101),
                           f6_t(0b100101),
                           f6_t(0b000110),
                           f6_t(0b100110),
                           f6_t(0b001011),
                           f6_t(0b101011)};
    // reference vector
    vector_type<f6x16_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<f6x16_pk_t>()(Number<0>{}).unpack(i), 0);
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<f6x16_pk_t>()(Number<i>{}) = f6x16_pk_t{test_vec};
    });

    // copy the vector
    vector_type<f6x16_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f6x16_pk_t>()(Number<0>{}).unpack(i),
                  static_cast<f6_t>(test_vec[static_cast<int>(i)]))
            << " i = " << i << "; left = "
            << type_convert<float>(left_vec.template AsType<f6x16_pk_t>()(Number<0>{}).unpack(i))
            << " -- right = "
            << type_convert<float>(static_cast<f6_t>(test_vec[static_cast<int>(i)])) << " ("
            << static_cast<int>(test_vec[static_cast<int>(i)]) << ")" << std::endl;
    });
}

__global__ void test_f6_convert_rne(float* p_test, uint64_t* p_completed)
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

    auto f6x32_vec = f6_convert_rne(float32_in);
    float32_out    = type_convert<ck::float32_t>(f6x32_vec);

    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = float32_out[static_cast<int>(ii)]; });
    i = N;
}

TEST(MXFP6, DeviceF6ConvertRNE)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_f6_convert_rne<<<1, 1>>>(static_cast<float*>(device_out.GetDeviceBuffer()),
                                  static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    EXPECT_EQ(N, completed);
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { EXPECT_EQ(out[static_cast<int>(ii)], 1.0f) << "ii: " << ii << std::endl; });

    auto f6x32_vec_tc    = ck::type_convert<f6x32_pk_t>(ck::float32_t(1.0f));
    auto f6x32_vec_cnstr = f6x32_pk_t(0x08);

    EXPECT_EQ(f6x32_vec_tc, f6x32_vec_cnstr);
}

// test vector of 2 f6x16_pk_t, contains 32 f6_t
TEST(FP6, TestAsType16x2)
{
    // test size
    const int vector_size = 2;
    const int packed_size = 16;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(16)));
    test_vec_t test_vec[2];
    test_vec[0] = {f6_t(0b000000),
                   f6_t(0b100000),
                   f6_t(0b000001),
                   f6_t(0b100001),
                   f6_t(0b000010),
                   f6_t(0b100010),
                   f6_t(0b000011),
                   f6_t(0b100011),
                   f6_t(0b000100),
                   f6_t(0b100100),
                   f6_t(0b000101),
                   f6_t(0b100101),
                   f6_t(0b000110),
                   f6_t(0b100110),
                   f6_t(0b001011),
                   f6_t(0b101011)};
    test_vec[1] = {f6_t(0b010000),
                   f6_t(0b110000),
                   f6_t(0b010001),
                   f6_t(0b110001),
                   f6_t(0b010010),
                   f6_t(0b110010),
                   f6_t(0b010011),
                   f6_t(0b110011),
                   f6_t(0b010100),
                   f6_t(0b110100),
                   f6_t(0b010101),
                   f6_t(0b110101),
                   f6_t(0b010110),
                   f6_t(0b110110),
                   f6_t(0b011011),
                   f6_t(0b111011)};
    // reference vector
    vector_type<f6x16_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, vector_size, 1>{}([&](auto idx_vector) {
        ck::static_for<0, packed_size, 1>{}([&](auto idx_element) {
            ASSERT_EQ(
                right_vec.template AsType<f6x16_pk_t>()(Number<idx_vector>{}).unpack(idx_element),
                0);
        });
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<f6x16_pk_t>()(Number<i>{}) = f6x16_pk_t{test_vec[i]};
    });
    // copy the vector
    vector_type<f6x16_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, vector_size, 1>{}([&](auto idx_vector) {
        ck::static_for<0, packed_size, 1>{}([&](auto idx_element) {
            ASSERT_EQ(
                left_vec.template AsType<f6x16_pk_t>()(Number<idx_vector>{}).unpack(idx_element),
                static_cast<f6_t>(test_vec[idx_vector][static_cast<int>(idx_element)]));
        });
    });
}

// test vector of 1 f6x32_pk_t, contains 32 f6_t
TEST(FP6, TestAsType32x1)
{
    // test size
    const int vector_size = 1;
    const int packed_size = 32;
    typedef int8_t test_vec_t __attribute__((ext_vector_type(32)));
    test_vec_t test_vec = {f6_t(0b000000), f6_t(0b100000), f6_t(0b000001), f6_t(0b100001),
                           f6_t(0b000010), f6_t(0b100010), f6_t(0b000011), f6_t(0b100011),
                           f6_t(0b000100), f6_t(0b100100), f6_t(0b000101), f6_t(0b100101),
                           f6_t(0b000110), f6_t(0b100110), f6_t(0b001011), f6_t(0b101011),
                           f6_t(0b010000), f6_t(0b110000), f6_t(0b010001), f6_t(0b110001),
                           f6_t(0b010010), f6_t(0b110010), f6_t(0b010011), f6_t(0b110011),
                           f6_t(0b010100), f6_t(0b110100), f6_t(0b010101), f6_t(0b110101),
                           f6_t(0b010110), f6_t(0b110110), f6_t(0b011011), f6_t(0b111011)};
    // reference vector
    vector_type<f6x32_pk_t, vector_size> right_vec;
    // check default CTOR
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(right_vec.template AsType<f6x32_pk_t>()(Number<0>{}).unpack(i), 0);
    });
    // assign test values to the vector
    ck::static_for<0, vector_size, 1>{}([&](auto i) {
        right_vec.template AsType<f6x32_pk_t>()(Number<i>{}) = f6x32_pk_t{test_vec};
    });
    // copy the vector
    vector_type<f6x32_pk_t, vector_size> left_vec{right_vec};
    // check if values were copied correctly
    ck::static_for<0, packed_size, 1>{}([&](auto i) {
        ASSERT_EQ(left_vec.template AsType<f6x32_pk_t>()(Number<0>{}).unpack(i),
                  static_cast<f6_t>(test_vec[static_cast<int>(i)]));
    });
}

TEST(FP6, TestAllValues)
{
    constexpr std::array<float, 64> e2m3ValuesOCP = {
        // clang-format off
        0.0000000000, 0.1250000000, 0.2500000000, 0.3750000000, 0.5000000000, 0.6250000000, 0.7500000000, 0.8750000000,
        1.0000000000, 1.1250000000, 1.2500000000, 1.3750000000, 1.5000000000, 1.6250000000, 1.7500000000, 1.8750000000,
        2.0000000000, 2.2500000000, 2.5000000000, 2.7500000000, 3.0000000000, 3.2500000000, 3.5000000000, 3.7500000000,
        4.0000000000, 4.5000000000, 5.0000000000, 5.5000000000, 6.0000000000, 6.5000000000, 7.0000000000, 7.5000000000,
        -0.0000000000, -0.1250000000, -0.2500000000, -0.3750000000, -0.5000000000, -0.6250000000, -0.7500000000, -0.8750000000,
        -1.0000000000, -1.1250000000, -1.2500000000, -1.3750000000, -1.5000000000, -1.6250000000, -1.7500000000, -1.8750000000,
        -2.0000000000, -2.2500000000, -2.5000000000, -2.7500000000, -3.0000000000, -3.2500000000, -3.5000000000, -3.7500000000,
        -4.0000000000, -4.5000000000, -5.0000000000, -5.5000000000, -6.0000000000, -6.5000000000, -7.0000000000, -7.5000000000
        // clang-format on
    };

    constexpr uint8_t e2m3BitsOCP[] = {
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
        printf("FP6 Table\n");
    ck::static_for<0, 64, 1>{}([&](auto i) {
        float fp = type_convert<float>(f6_t(e2m3BitsOCP[i]));
        ASSERT_EQ(fp, e2m3ValuesOCP[i]);

        f6_t fp6 = type_convert<f6_t>(e2m3ValuesOCP[i]);
        ASSERT_EQ(fp6 & 0x3F, e2m3BitsOCP[i] & 0x3F);

        if(ck_logging)
        {
            // Print the binary representation
            printf("Bits: 0b");
            for(int j = 5; j >= 0; --j)
            {
                printf("%c", (e2m3BitsOCP[i] & (1 << j)) ? '1' : '0');
            }
            printf(", 0x%02X, Value: %f\n", e2m3BitsOCP[i], e2m3ValuesOCP[i]);
        }
    });
}
