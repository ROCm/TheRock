// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"

using ::ck::DeviceMem;

using ck::bf8_ocp_t;
using ck::bf8x2_ocp_t;
using ck::bhalf2_t;
using ck::bhalf_t;
using ck::f8_convert_rne;
using ck::f8_convert_sr;
using ck::float2_t;
using ck::half2_t;
using ck::half_t;
using ck::type_convert;

TEST(BF8OCP, NumericLimits)
{ // constants given for OCP FP8
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::Min(),
              type_convert<bf8_ocp_t>(0x04)); // 0b00000100 = 2^-14
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::Max(),
              type_convert<bf8_ocp_t>(0x7B)); // 0b01111011 = 57344
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::Lowest(),
              type_convert<bf8_ocp_t>(0xFB)); // 0b11111011 = -57344
    EXPECT_EQ(ck::NumericLimits<bf8_ocp_t>::QuietNaN().data,
              type_convert<bf8_ocp_t>(0x7D).data); // 0b01111101
    EXPECT_FALSE(ck::NumericLimits<bf8_ocp_t>::QuietNaN() ==
                 ck::NumericLimits<bf8_ocp_t>::QuietNaN());
    EXPECT_TRUE(ck::fp8_is_inf(type_convert<bf8_ocp_t>(0xFC)) &&
                ck::fp8_is_inf(type_convert<bf8_ocp_t>(0x7C)));
}

TEST(BF8OCP, ConvertFP32Nearest)
{
    // fix the tolerance value
    float abs_tol = 1e-6;

    // convert 0 float to bfp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_rne<bf8_ocp_t>(0.0f)), 0.0f);

    // convert minimal float to bf8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::min())),
                abs_tol);

    const auto max_bf8_t_float = type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to float and check if equal to bf8 max
    ASSERT_NEAR(
        max_bf8_t_float, type_convert<float>(f8_convert_rne<bf8_ocp_t>(max_bf8_t_float)), 0.0f);

    // convert maximal float to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_float,
                type_convert<float>(f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::max())),
                0.0f);

    // convert float infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(ck::NumericLimits<bf8_ocp_t>::Max(),
              f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::infinity()));

    // positive normal float value to bf8 and back, check if holds
    float pos_float = 0.0000762939f; // 10*2^-17
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_rne<bf8_ocp_t>(pos_float)), abs_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr auto neg_min_bf8 = -0.00006103515625f; //-2^-14
    ASSERT_NEAR(neg_min_bf8, type_convert<float>(f8_convert_rne<bf8_ocp_t>(neg_min_bf8)), 0.0f);

    // positive subnorm float value to bf8 and back, check if holds
    constexpr auto pos_subnorm_bf8 = 0.000030517578125f; // 2^-15
    ASSERT_NEAR(
        pos_subnorm_bf8, type_convert<float>(f8_convert_rne<bf8_ocp_t>(pos_subnorm_bf8)), 0.0f);

    // min subnorm bf8 value to bf8 and back, check if holds
    constexpr auto min_subnorm_bf8 = -0.0000152587890625f; //-2^-16
    ASSERT_NEAR(
        min_subnorm_bf8, type_convert<float>(f8_convert_rne<bf8_ocp_t>(min_subnorm_bf8)), 0.0f);

    // smaller than min subnorm bf8 value to bf8 must be zero
    constexpr auto less_than_min_subnorm = 0.00000762939453125f; // 2^-17
    ASSERT_EQ(0.0f, type_convert<float>(f8_convert_rne<bf8_ocp_t>(less_than_min_subnorm)));

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}

TEST(BF8OCP, ConvertFP32Stochastic)
{
    // fix the tolerance value
    float abs_tol = 1e-6;

    // convert 0 float to bfp8 and back, check if holds
    ASSERT_NEAR(0.0f, type_convert<float>(f8_convert_sr<bf8_ocp_t>(0.0f)), 0.0f);

    // convert minimal float to bf8 and back, check if holds
    ASSERT_NEAR(std::numeric_limits<float>::min(),
                type_convert<float>(f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::min())),
                abs_tol);

    const auto max_bf8_t_float = type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to float and check if equal to bf8 max
    ASSERT_NEAR(
        max_bf8_t_float, type_convert<float>(f8_convert_sr<bf8_ocp_t>(max_bf8_t_float)), 0.0f);

    // convert maximal float to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_float,
                type_convert<float>(f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::max())),
                0.0f);

    // convert float infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(ck::NumericLimits<bf8_ocp_t>::Max(),
              f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::infinity()));

    // positive normal float value to bf8 and back, check if holds
    float pos_float = 0.0000762939f; // 10*2^-17
    ASSERT_NEAR(pos_float, type_convert<float>(f8_convert_sr<bf8_ocp_t>(pos_float)), abs_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr auto neg_min_bf8 = -0.00006103515625f; //-2^-14
    ASSERT_NEAR(neg_min_bf8, type_convert<float>(f8_convert_sr<bf8_ocp_t>(neg_min_bf8)), 0.0f);

    // positive subnorm float value to bf8 and back, check if holds
    constexpr auto pos_subnorm_bf8 = 0.000030517578125f; // 2^-15
    ASSERT_NEAR(
        pos_subnorm_bf8, type_convert<float>(f8_convert_sr<bf8_ocp_t>(pos_subnorm_bf8)), 0.0f);

    // min subnorm bf8 value to bf8 and back, check if holds
    constexpr auto min_subnorm_bf8 = -0.0000152587890625f; //-2^-16
    ASSERT_NEAR(
        min_subnorm_bf8, type_convert<float>(f8_convert_sr<bf8_ocp_t>(min_subnorm_bf8)), 0.0f);

    // smaller than min subnorm bf8 value to bf8  alternates between 0 and 2^-16
    constexpr auto less_than_min_subnorm = 0.00000762939453125f; // 2^-17
    ASSERT_NEAR(0.0f,
                type_convert<float>(f8_convert_sr<bf8_ocp_t>(less_than_min_subnorm)),
                0.0000152587890625f);

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_sr<bf8_ocp_t>(std::numeric_limits<float>::quiet_NaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}

TEST(BF8OCP, ConvertFP16Nearest)
{
    // fix the tolerance value
    constexpr half_t half_t_tol  = 1e-3;
    constexpr half_t half_t_zero = 0.0;

    // convert 0 half_t to bfp8 and back, check if holds
    ASSERT_NEAR(
        half_t_zero, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(half_t_zero)), half_t_zero);

    // convert minimal half_t to bf8 and back, check if holds
    ASSERT_NEAR(ck::NumericLimits<half_t>::Min(),
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(ck::NumericLimits<half_t>::Min())),
                half_t_tol);

    const auto max_bf8_t_half_t = type_convert<half_t>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to half_t and check if equal to bf8 max
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(max_bf8_t_half_t)),
                half_t_zero);

    // convert maximal half_t to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(ck::NumericLimits<half_t>::Max())),
                half_t_zero);

    // convert half_t infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(
        ck::NumericLimits<bf8_ocp_t>::Max(),
        f8_convert_rne<bf8_ocp_t>(type_convert<half_t>(std::numeric_limits<float>::infinity())));

    // positive normal bf8 value to bf8 and back, check if holds
    constexpr half_t pos_norm_bf8{0.0000762939f}; // 10*2^-17
    ASSERT_NEAR(
        pos_norm_bf8, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(pos_norm_bf8)), half_t_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr half_t neg_min_bf8{-0.00006103515625f}; //-2^-14
    ASSERT_NEAR(
        neg_min_bf8, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(neg_min_bf8)), half_t_zero);

    // positive subnorm bf8 value to bf8 and back, check if holds
    constexpr half_t pos_subnorm_bf8{0.000030517578125f}; // 2^-15
    ASSERT_NEAR(pos_subnorm_bf8,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(pos_subnorm_bf8)),
                half_t_zero);

    // min subnorm bf8 value to bf8 and back, check if holds
    constexpr half_t min_subnorm_bf8{-0.0000152587890625f}; //-2^-16
    ASSERT_NEAR(min_subnorm_bf8,
                type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(min_subnorm_bf8)),
                half_t_zero);

    // smaller than min subnorm bf8 value to bf8 must be zero
    constexpr half_t less_than_min_subnorm{0.00000762939453125f}; // 2^-17
    ASSERT_EQ(half_t_zero, type_convert<half_t>(f8_convert_rne<bf8_ocp_t>(less_than_min_subnorm)));

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_rne<bf8_ocp_t>(ck::NumericLimits<half_t>::QuietNaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}

TEST(BF8OCP, ConvertFP16Stochastic)
{
    // fix the tolerance value
    constexpr half_t half_t_tol    = 1e-3;
    constexpr half_t half_t_zero   = 0.0;
    constexpr auto min_subnorm_bf8 = 0.0000152587890625f; // 2^-16

    // convert 0 half_t to bfp8 and back, check if holds
    ASSERT_NEAR(
        half_t_zero, type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(half_t_zero)), half_t_zero);

    // convert minimal half_t (6.103515625e-05) to fp8 and back
    ASSERT_NEAR(ck::NumericLimits<half_t>::Min(),
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(ck::NumericLimits<half_t>::Min())),
                half_t_zero);

    const auto max_bf8_t_half_t = type_convert<half_t>(ck::NumericLimits<bf8_ocp_t>::Max());

    // convert maximal bf8_ocp_t to half_t and check if equal to bf8 max
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(max_bf8_t_half_t)),
                half_t_zero);

    // convert maximal half_t to bf8 and back, check if clipped to bf8 max (saturation to finite)
    ASSERT_NEAR(max_bf8_t_half_t,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(ck::NumericLimits<half_t>::Max())),
                half_t_zero);

    // convert half_t infinity to bf8_ocp_t and check if it is max value (saturation to finite)
    ASSERT_EQ(
        ck::NumericLimits<bf8_ocp_t>::Max(),
        f8_convert_sr<bf8_ocp_t>(type_convert<half_t>(std::numeric_limits<float>::infinity())));

    // positive normal bf8 value to bf8 and back, check if holds
    constexpr half_t pos_norm_bf8{0.0000762939f}; // 10*2^-17
    ASSERT_NEAR(
        pos_norm_bf8, type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(pos_norm_bf8)), half_t_tol);

    // negative smallest normal bf8 value to bf8 and back, check if holds
    constexpr half_t neg_min_bf8{-0.00006103515625f}; //-2^-14
    ASSERT_NEAR(
        neg_min_bf8, type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(neg_min_bf8)), half_t_zero);

    // positive subnorm bf8 value to bf8 and back, check if holds
    constexpr half_t pos_subnorm_bf8{0.000030517578125f}; // 2^-15
    ASSERT_NEAR(pos_subnorm_bf8,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(pos_subnorm_bf8)),
                half_t_zero);

    // min subnorm bf8 value to bf8 and back, check if holds
    ASSERT_NEAR(half_t{-min_subnorm_bf8},
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(half_t{-min_subnorm_bf8})),
                half_t_zero);

    // smaller than min subnorm bf8 value to bf8  alternates between 0 and 2^-16
    constexpr half_t less_than_min_subnorm{0.00000762939453125f}; // 2^-17
    ASSERT_NEAR(half_t_zero,
                type_convert<half_t>(f8_convert_sr<bf8_ocp_t>(less_than_min_subnorm)),
                half_t{min_subnorm_bf8});

    // convert quiet NaN to bf8_ocp_t and check if it is quiet NaN
    const auto bf8_nan = f8_convert_sr<bf8_ocp_t>(ck::NumericLimits<half_t>::QuietNaN());
    ASSERT_TRUE(ck::fp8_impl::ocp_bf8_is_nan(bf8_nan.data));
}

constexpr uint64_t test_size = 256 + 6;

__host__ __device__ void
test_fp32_bf8_type_convert(uint64_t N, float* p_test, uint64_t* p_completed)
{
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

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto v          = type_convert<float>(bf8_ocp_t{bf8_uid});
        p_test[i]       = v;
        i++;
        if(i >= N)
        {
            return;
        }
    }

    /// Test vector conversion
    // bf8x2 -> fp32x2
    bf8x2_ocp_t bf8x2{bf8x2_ocp_t::data_v{0b10000100, 0b00000001}}; //-2^-14, 2^-16

    float2_t f32x2 = type_convert<float2_t>(bf8x2);
    p_test[i++]    = f32x2[0];
    if(i >= N)
    {
        return;
    }
    p_test[i++] = f32x2[1];
    if(i >= N)
    {
        return;
    }

    // fp32x2 -> bf8x2
    f32x2 = {-4.0f, 2.0f};
    bf8x2 = f8_convert_rne<bf8x2_ocp_t>(f32x2); // expect {-4, 2}

    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }

    bf8x2 = f8_convert_sr<bf8x2_ocp_t>(f32x2); // expect {-4, 2}

    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }
}

TEST(BF8OCP, HostFP32BF8Convert)
{
    std::vector<float> out(test_size, -1.0f);
    uint64_t completed = 0;

    test_fp32_bf8_type_convert(test_size, out.data(), &completed);

    std::set<uint8_t> bf8_nan_ids;
    bf8_nan_ids.insert(0b11111111);
    bf8_nan_ids.insert(0b01111111);
    bf8_nan_ids.insert(0b11111101);
    bf8_nan_ids.insert(0b01111101);
    bf8_nan_ids.insert(0b11111110);
    bf8_nan_ids.insert(0b01111110);
    for(auto bf8_nan_id : bf8_nan_ids)
    {
        auto idx = bf8_nan_id;
        ASSERT_TRUE(std::isnan(out[idx]));
    }

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        if(bf8_nan_ids.find(bf8_id) != bf8_nan_ids.end())
            continue;

        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto idx        = bf8_uid;
        ASSERT_FLOAT_EQ(out[idx], type_convert<float>(bf8_ocp_t{bf8_uid}))
            << " bf8_id: " << bf8_id << std::endl
            << type_convert<float>(bf8_ocp_t{bf8_uid});
    }

    // /// Test vector conversions

    auto i = 256;

    // bf8x2 -> fp32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -14.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -16.0f));

    // fp32x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);
    // SR
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__global__ void device_test_fp32_bf8_type_convert(uint64_t N, float* p_test, uint64_t* p_completed)
{
    test_fp32_bf8_type_convert(N, p_test, p_completed);
}

TEST(BF8OCP, DeviceFP32BF8Convert)
{
    std::vector<float> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    device_test_fp32_bf8_type_convert<<<1, 1>>>(
        test_size,
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    std::set<uint8_t> bf8_nan_ids;
    bf8_nan_ids.insert(0b11111111);
    bf8_nan_ids.insert(0b01111111);
    bf8_nan_ids.insert(0b11111101);
    bf8_nan_ids.insert(0b01111101);
    bf8_nan_ids.insert(0b11111110);
    bf8_nan_ids.insert(0b01111110);
    for(auto bf8_nan_id : bf8_nan_ids)
    {
        auto idx = bf8_nan_id;
        ASSERT_TRUE(std::isnan(out[idx])) << "idx: " << idx << " out[idx]: " << out[idx];
    }

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        if(bf8_nan_ids.find(bf8_id) != bf8_nan_ids.end())
            continue;

        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto idx        = bf8_uid;
        ASSERT_FLOAT_EQ(out[idx], type_convert<float>(bf8_ocp_t{bf8_uid}))
            << " bf8_id: " << bf8_id << std::endl
            << type_convert<float>(bf8_ocp_t{bf8_uid});
    }

    /// Test vector conversions

    auto i = 256;

    // bf8x2 -> fp32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -14.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -16.0f));

    // fp32x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);
    // SR
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__host__ __device__ void
test_fp16_bf8_type_convert(uint64_t N, half_t* p_test, uint64_t* p_completed)
{
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

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto v          = type_convert<half_t>(bf8_ocp_t{bf8_uid});
        p_test[i]       = v;
        i++;
        if(i >= N)
        {
            return;
        }
    }

    /// Test vector conversion
    // bf8x2 -> fp16x2
    bf8x2_ocp_t bf8x2{bf8x2_ocp_t::data_v{0b10000100, 0b00000001}}; //-2^-14, 2^-16

    half2_t f16x2 = type_convert<half2_t>(bf8x2);
    p_test[i++]   = f16x2[0];
    if(i >= N)
    {
        return;
    }
    p_test[i++] = f16x2[1];
    if(i >= N)
    {
        return;
    }

    // fp16x2 -> bf8x2
    f16x2 = {-4.0f, 2.0f};
    bf8x2 = f8_convert_rne<bf8x2_ocp_t>(f16x2); // expect {-4, 2}

    p_test[i++] = type_convert<half_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<half_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }

    bf8x2 = f8_convert_sr<bf8x2_ocp_t>(f16x2); // expect {-4, 2}

    p_test[i++] = type_convert<half_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<half_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }
}

TEST(BF8OCP, HostFP16BF8Convert)
{
    std::vector<half_t> out(test_size, -1.0f);
    uint64_t completed = 0;

    test_fp16_bf8_type_convert(test_size, out.data(), &completed);

    std::set<uint8_t> bf8_nan_ids;
    bf8_nan_ids.insert(0b11111111);
    bf8_nan_ids.insert(0b01111111);
    bf8_nan_ids.insert(0b11111101);
    bf8_nan_ids.insert(0b01111101);
    bf8_nan_ids.insert(0b11111110);
    bf8_nan_ids.insert(0b01111110);
    for(auto bf8_nan_id : bf8_nan_ids)
    {
        auto idx = bf8_nan_id;
        ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])));
    }

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        if(bf8_nan_ids.find(bf8_id) != bf8_nan_ids.end())
            continue;

        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto idx        = bf8_uid;
        ASSERT_FLOAT_EQ(out[idx], type_convert<half_t>(bf8_ocp_t{bf8_uid}))
            << " bf8_id: " << bf8_id << std::endl
            << type_convert<float>(type_convert<half_t>(bf8_ocp_t{bf8_uid}));
    }

    // /// Test vector conversions

    auto i = 256;

    // bf8x2 -> fp16x2
    EXPECT_EQ(out[i++], type_convert<half_t>(-powf(2.0f, -14.0f)));
    EXPECT_EQ(out[i++], type_convert<half_t>(powf(2.0f, -16.0f)));

    // fp16x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], type_convert<half_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<half_t>(2.0f));
    // SR
    EXPECT_EQ(out[i++], type_convert<half_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<half_t>(2.0f));

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__global__ void device_test_fp16_bf8_type_convert(uint64_t N, half_t* p_test, uint64_t* p_completed)
{
    test_fp16_bf8_type_convert(N, p_test, p_completed);
}

TEST(BF8OCP, DeviceFP16BF8Convert)
{
    std::vector<half_t> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(half_t));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    device_test_fp16_bf8_type_convert<<<1, 1>>>(
        test_size,
        static_cast<half_t*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    std::set<uint8_t> bf8_nan_ids;
    bf8_nan_ids.insert(0b11111111);
    bf8_nan_ids.insert(0b01111111);
    bf8_nan_ids.insert(0b11111101);
    bf8_nan_ids.insert(0b01111101);
    bf8_nan_ids.insert(0b11111110);
    bf8_nan_ids.insert(0b01111110);
    for(auto bf8_nan_id : bf8_nan_ids)
    {
        auto idx = bf8_nan_id;
        ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])))
            << "idx: " << idx << " out[idx]: " << type_convert<float>(out[idx]);
    }

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        if(bf8_nan_ids.find(bf8_id) != bf8_nan_ids.end())
            continue;

        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto idx        = bf8_uid;
        ASSERT_FLOAT_EQ(out[idx], type_convert<half_t>(bf8_ocp_t{bf8_uid}))
            << " bf8_id: " << bf8_id << std::endl
            << type_convert<float>(type_convert<half_t>(bf8_ocp_t{bf8_uid}));
    }

    /// Test vector conversions

    auto i = 256;

    // bf8x2 -> fp16x2
    EXPECT_EQ(out[i++], type_convert<half_t>(-powf(2.0f, -14.0f)));
    EXPECT_EQ(out[i++], type_convert<half_t>(powf(2.0f, -16.0f)));

    // fp16x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], type_convert<half_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<half_t>(2.0f));
    // SR
    EXPECT_EQ(out[i++], type_convert<half_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<half_t>(2.0f));

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__host__ __device__ void
test_bf16_bf8_type_convert(uint64_t N, bhalf_t* p_test, uint64_t* p_completed)
{
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

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto v          = type_convert<bhalf_t>(bf8_ocp_t{bf8_uid});
        p_test[i]       = v;
        i++;
        if(i >= N)
        {
            return;
        }
    }

    /// Test vector conversion
    // bf8x2 -> bf16x2
    bf8x2_ocp_t bf8x2{bf8x2_ocp_t::data_v{0b10000100, 0b00000001}}; //-2^-14, 2^-16

    bhalf2_t bf16x2 = type_convert<bhalf2_t>(bf8x2);
    p_test[i++]     = bf16x2[0];
    if(i >= N)
    {
        return;
    }
    p_test[i++] = bf16x2[1];
    if(i >= N)
    {
        return;
    }

    // bf16x2 -> bf8x2
    bf16x2 = {type_convert<bhalf_t>(-4.0f), type_convert<bhalf_t>(2.0f)};
    bf8x2  = f8_convert_rne<bf8x2_ocp_t>(bf16x2); // expect {-4, 2}

    p_test[i++] = type_convert<bhalf_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<bhalf_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }

    bf8x2 = f8_convert_sr<bf8x2_ocp_t>(bf16x2); // expect {-4, 2}

    p_test[i++] = type_convert<bhalf_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<bhalf_t>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }
}

TEST(BF8OCP, HostBF16BF8Convert)
{
    std::vector<bhalf_t> out(test_size, -1.0f);
    uint64_t completed = 0;

    test_bf16_bf8_type_convert(test_size, out.data(), &completed);

    std::set<uint8_t> bf8_nan_ids;
    bf8_nan_ids.insert(0b11111111);
    bf8_nan_ids.insert(0b01111111);
    bf8_nan_ids.insert(0b11111101);
    bf8_nan_ids.insert(0b01111101);
    bf8_nan_ids.insert(0b11111110);
    bf8_nan_ids.insert(0b01111110);
    for(auto bf8_nan_id : bf8_nan_ids)
    {
        auto idx = bf8_nan_id;
        ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])));
    }

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        if(bf8_nan_ids.find(bf8_id) != bf8_nan_ids.end())
            continue;

        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto idx        = bf8_uid;
        ASSERT_FLOAT_EQ(out[idx], type_convert<bhalf_t>(bf8_ocp_t{bf8_uid}))
            << " bf8_id: " << bf8_id << std::endl
            << type_convert<float>(type_convert<bhalf_t>(bf8_ocp_t{bf8_uid}));
    }

    // /// Test vector conversions

    auto i = 256;

    // bf8x2 -> bf16x2
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(-powf(2.0f, -14.0f)));
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(powf(2.0f, -16.0f)));

    // bf16x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(2.0f));
    // SR
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(2.0f));

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__global__ void
device_test_bf16_bf8_type_convert(uint64_t N, bhalf_t* p_test, uint64_t* p_completed)
{
    test_bf16_bf8_type_convert(N, p_test, p_completed);
}

TEST(BF8OCP, DeviceBF16BF8Convert)
{
    std::vector<bhalf_t> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(bhalf_t));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    device_test_bf16_bf8_type_convert<<<1, 1>>>(
        test_size,
        static_cast<bhalf_t*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    std::set<uint8_t> bf8_nan_ids;
    bf8_nan_ids.insert(0b11111111);
    bf8_nan_ids.insert(0b01111111);
    bf8_nan_ids.insert(0b11111101);
    bf8_nan_ids.insert(0b01111101);
    bf8_nan_ids.insert(0b11111110);
    bf8_nan_ids.insert(0b01111110);
    for(auto bf8_nan_id : bf8_nan_ids)
    {
        auto idx = bf8_nan_id;
        ASSERT_TRUE(std::isnan(type_convert<float>(out[idx])))
            << "idx: " << idx << " out[idx]: " << type_convert<float>(out[idx]);
    }

    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        if(bf8_nan_ids.find(bf8_id) != bf8_nan_ids.end())
            continue;

        uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
        auto idx        = bf8_uid;
        ASSERT_FLOAT_EQ(out[idx], type_convert<bhalf_t>(bf8_ocp_t{bf8_uid}))
            << " bf8_id: " << bf8_id << std::endl
            << type_convert<float>(type_convert<bhalf_t>(bf8_ocp_t{bf8_uid}));
    }

    /// Test vector conversions

    auto i = 256;

    // bf8x2 -> bf16x2
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(-powf(2.0f, -14.0f)));
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(powf(2.0f, -16.0f)));

    // bf16x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(2.0f));
    // SR
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(-4.0f));
    EXPECT_EQ(out[i++], type_convert<bhalf_t>(2.0f));

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}
