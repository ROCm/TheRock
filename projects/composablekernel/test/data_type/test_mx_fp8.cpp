// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ck::e8m0_bexp_t;
using ck::f8_ocp_t;
using ck::f8x16_ocp_t;
using ck::f8x2_ocp_t;
using ck::f8x32_ocp_t;
using ck::float16_t;
using ck::float2_t;
using ck::float32_t;
using ck::mxf8_convert_rne;
using ck::mxf8_convert_sr;
using ck::scaled_type_convert;
using ck::type_convert;
using ck::fp8_impl::fp8x2_storage_t;

using ::ck::DeviceMem;

constexpr uint64_t test_size = 256 * 256 + 2 + 4 + 6;

/**
 * @brief Tests conversion of FP8 values to float using E8M0 exponent scaling.
 *
 * This function performs a series of conversions from FP8 values to float values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 and FP8 values,
 * as well as specific vector and rounding conversions.
 *
 * @param N The maximum number of conversions to perform.
 * @param p_test Pointer to the output array where the converted float values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 * @note If either p_test or p_completed is nullptr, the function will return immediately.
 * @note The function will stop converting if the number of conversions reaches N.
 * @note First 256*256 conversions are for all possible combinations of E8M0 and FP8 values that are
 * stored in memory sequentially with FP8 values varying faster.
 *
 * The function performs the following conversions:
 * - All possible combinations of E8M0 and FP8 values. [256x256]
 * - Vector conversions f8x2 -> f32x2. [2]
 * - Vector conversions  f32x2 -> f8x2 rne. [2]
 * - Vector conversions  f32x2 -> f8x2 sr. [2]
 * - Round to nearest even conversions for specific float values. [6]
 *
 * The results are stored in the p_test array, and the number of completed conversions
 * is updated in the p_completed variable.
 */
__host__ __device__ void
test_mx_fp8_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
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

    // All possible combinations of E8M0 and FP8
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        for(ck::index_t fp8_id = 0; fp8_id < 256; fp8_id++)
        {
            uint8_t fp8_uid = static_cast<uint8_t>(fp8_id);
            auto v          = scaled_type_convert<float>(e8m0_bexp_t(exp_id), f8_ocp_t{fp8_uid});
            p_test[i]       = v;
            i++;
            if(i >= N)
            {
                return;
            }
        }
    }

    /// Test vector conversions
    // f8x2 -> f32x2
    f8x2_ocp_t fp8x2{f8x2_ocp_t::data_v{0b10001000, 0b00000001}}; //-2^-6, 2^-9
    auto scale2 = e8m0_bexp_t(2.0f);

    float2_t f32x2 = scaled_type_convert<float2_t>(scale2, fp8x2);
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

    // f32x2 -> f8x2
    f32x2 = {-8.0f, 4.0f};
    fp8x2 = mxf8_convert_rne<f8x2_ocp_t>(f32x2, type_convert<float>(scale2)); // expect {-4, 2}

    p_test[i++] = type_convert<float>(fp8x2.AsType<f8_ocp_t>()(ck::Number<0>{})); //-4f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(fp8x2.AsType<f8_ocp_t>()(ck::Number<1>{})); // 2f
    if(i >= N)
    {
        return;
    }

    auto scale4 = e8m0_bexp_t(4.0f);

    fp8x2 = mxf8_convert_sr<f8x2_ocp_t>(f32x2, type_convert<float>(scale4)); // expect {-2, 1}

    p_test[i++] = type_convert<float>(fp8x2.AsType<f8_ocp_t>()(ck::Number<0>{})); //-2f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(fp8x2.AsType<f8_ocp_t>()(ck::Number<1>{})); // 1f
    if(i >= N)
    {
        return;
    }

    /// Test round to nearest even

    p_test[i++] = type_convert<float>(mxf8_convert_rne<f8_ocp_t>(1024.0f, 4.0f)); // 1024/4
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(
        mxf8_convert_rne<f8_ocp_t>(std::numeric_limits<float>::quiet_NaN(), 4.0f)); // => NaN
    if(i >= N)
    {
        return;
    }

    // Inf/2 > 448 => NaN on device
    p_test[i++] = type_convert<float>(
        mxf8_convert_rne<f8_ocp_t>(std::numeric_limits<float>::infinity(), 2.0f));
    if(i >= N)
    {
        return;
    }

    // 256/0.5  > 448 => NaN on device
    p_test[i++] = type_convert<float>(mxf8_convert_rne<f8_ocp_t>(256.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    // -256/0.5  < -448 => NaN on device
    p_test[i++] = type_convert<float>(mxf8_convert_rne<f8_ocp_t>(-256.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    // proper scale selection 2^13 < 10000; 2^8 < 448 => scale = 2^(13-8) = 2^5
    p_test[i++] =
        type_convert<float>(mxf8_convert_rne<f8_ocp_t>(10000.0f, 32.0f)); // 10000/32 = 312.5
    if(i >= N)
    {
        return;
    }
}

TEST(MXFP8, HostScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);
    uint64_t completed = 0;

    test_mx_fp8_scaled_convert(test_size, out.data(), &completed);

    // V = X * P; X - E8M0 scale, P - FP8

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t fp8_id = 0; fp8_id < 256; fp8_id++)
    {
        auto idx = e8m0_nan_id * 256 + fp8_id;
        ASSERT_TRUE(std::isnan(out[idx]));
    }

    // If P in {Inf, NaN}, then V = P
    std::set<uint8_t> fp8_nan_ids;
    fp8_nan_ids.insert(0b11111111); //-NaN
    fp8_nan_ids.insert(0b01111111); // +NaN
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(auto fp8_nan_id : fp8_nan_ids)
        {
            auto idx = exp_id * 256 + fp8_nan_id;
            ASSERT_TRUE(std::isnan(out[idx]));
        }
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t fp8_id = 0; fp8_id < 256; fp8_id++)
        {
            if(fp8_nan_ids.find(fp8_id) != fp8_nan_ids.end())
                continue;

            uint8_t fp8_uid = static_cast<uint8_t>(fp8_id);
            auto idx        = exp_id * 256 + fp8_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(f8_ocp_t{fp8_uid}))
                << "exp_id: " << exp_id << " fp8_id: " << fp8_id << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(f8_ocp_t{fp8_uid});
        }
    }

    /// Test vector conversions

    auto i = 256 * 256;

    // f8x2 -> f32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -5.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -8.0f));

    // f32x2 -> fp8x2
    // RNE
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);
    // SR
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 1.0f);

    /// Test round to nearest even
    EXPECT_EQ(out[i++], 1024.0f / 4.0f) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f8_ocp_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(type_convert<f8_ocp_t>(312.5f)))
        << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__global__ void test_mx_fp8_device_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
{
    test_mx_fp8_scaled_convert(N, p_test, p_completed);
}

TEST(MXFP8, DeviceScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp8_device_scaled_convert<<<1, 1>>>(
        test_size,
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    // V = X * P; X - E8M0 scale, P - FP8

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t fp8_id = 0; fp8_id < 256; fp8_id++)
    {
        auto idx = e8m0_nan_id * 256 + fp8_id;
        ASSERT_TRUE(std::isnan(out[idx])) << "idx: " << idx << " out[idx]: " << out[idx];
    }

    // If P in {Inf, NaN}, then V = P
    std::set<uint8_t> fp8_nan_ids;
    fp8_nan_ids.insert(0b11111111); //-NaN
    fp8_nan_ids.insert(0b01111111); // +NaN
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(auto fp8_nan_id : fp8_nan_ids)
        {
            auto idx = exp_id * 256 + fp8_nan_id;
            ASSERT_TRUE(std::isnan(out[idx])) << "idx: " << idx << " out[idx]: " << out[idx];
        }
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t fp8_id = 0; fp8_id < 256; fp8_id++)
        {
            if(fp8_nan_ids.find(fp8_id) != fp8_nan_ids.end())
                continue;

            uint8_t fp8_uid = static_cast<uint8_t>(fp8_id);
            auto idx        = exp_id * 256 + fp8_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(f8_ocp_t{fp8_uid}))
                << "exp_id: " << exp_id << " fp8_id: " << fp8_id << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(f8_ocp_t{fp8_uid});
        }
    }

    /// Test vector conversions

    auto i = 256 * 256;

    // f8x2 -> f32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -5.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -8.0f));

    // f32x2 -> fp8x2
    // RNE
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);
    // SR
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 1.0f);

    /// Test round to nearest even
    EXPECT_EQ(out[i++], 1024.0f / 4.0f) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
#if 1
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];
#else
    // NOTE: Host and Device have different behavior.
    // Device returns NaN, while Host returns Max (saturation to finite value).
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f8_ocp_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
#endif
    EXPECT_EQ(out[i++], type_convert<float>(type_convert<f8_ocp_t>(312.5f)))
        << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__host__ __device__ float vec16_generator(ck::index_t i)
{
    return (i < 8 ? -1.0 : 1.0) * powf(2.0f, i % 8);
}

__global__ void test_mx_fp8x16_device_scaled_convert(float* p_test, uint64_t* p_completed)
{
    constexpr int N = 16;
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

    auto scale2 = e8m0_bexp_t(2.0f);

    f8x16_ocp_t fp8x16{};
    float16_t float16{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float16[static_cast<int>(ii)] = vec16_generator(ii); });

    fp8x16 = scaled_type_convert<ck::f8x16_ocp_t>(scale2, float16);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(fp8x16.AsType<f8_ocp_t>()(ck::Number<ii>{}));
    });
}

TEST(MXFP8, DeviceF32x16ToF8x16ScaledConvert)
{
    constexpr int N = 16;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp8x16_device_scaled_convert<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec16_generator(ii) / 2.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

__host__ __device__ float vec32_generator(ck::index_t i)
{
    if(i < 16)
    {
        return vec16_generator(i % 16);
    }
    else
    {
        return 1.5f * vec16_generator(i % 16);
    }
}

__global__ void test_mx_fp8x32_device_scaled_convert(float* p_test, uint64_t* p_completed)
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

    auto scale2 = e8m0_bexp_t(2.0f);

    f8x32_ocp_t fp8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float32[static_cast<int>(ii)] = vec32_generator(ii); });

    fp8x32 = mxf8_convert_rne<f8x32_ocp_t>(float32, type_convert<float>(scale2));

    ck::static_for<0, N, 1>{}(
        [&](auto ii) { p_test[i++] = type_convert<float>(fp8x32.AsType<f8_ocp_t>()(ii)); });
}

TEST(MXFP8, DeviceF32x32ToF8x32ScaledConvert)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp8x32_device_scaled_convert<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec32_generator(ii) / 2.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

__global__ void test_mx_fp8x32_device_scaled_convert_sr(float* p_test, uint64_t* p_completed)
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

    auto scale2 = e8m0_bexp_t(8.0f);

    f8x32_ocp_t fp8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float32[static_cast<int>(ii)] = vec32_generator(ii); });

    fp8x32 = mxf8_convert_sr<f8x32_ocp_t>(float32, type_convert<float>(scale2));

    ck::static_for<0, N, 1>{}(
        [&](auto ii) { p_test[i++] = type_convert<float>(fp8x32.AsType<f8_ocp_t>()(ii)); });
}

TEST(MXFP8, DeviceF32x32ToF8x32ScaledConvertSR)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp8x32_device_scaled_convert_sr<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec32_generator(ii) / 8.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

__global__ void test_mx_f32x32_device_scaled_convert(float* p_test, uint64_t* p_completed)
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

    auto scale2 = e8m0_bexp_t(4.0f);

    f8x32_ocp_t fp8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}([&](auto ii) {
        fp8x32.AsType<f8_ocp_t>()(ii) = type_convert<f8_ocp_t>(vec32_generator(ii) / 16.0f);
    });

    float32 = scaled_type_convert<float32_t>(scale2, fp8x32);

    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = float32[static_cast<int>(ii)]; });
}

TEST(MXFP8, DeviceF8x32ToF32x32ScaledConvert)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_f32x32_device_scaled_convert<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i = 0;

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec32_generator(ii) / 4.0f) << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}
