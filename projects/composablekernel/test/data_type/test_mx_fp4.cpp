// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ::ck::DeviceMem;

using ck::e8m0_bexp_t;
using ck::float16_t;
using ck::float2_t;
using ck::float32_t;
using ck::scaled_type_convert;
using ck::type_convert;

using ck::f4_convert_rne;
using ck::f4_convert_sr;
using ck::f4_t;
using ck::f4x16_t;
using ck::f4x2_pk_t;
using ck::f4x2_t;
using ck::f4x32_t;

constexpr uint64_t test_size = 256 * 16 + 2 + 4 + 6;

/**
 * @brief Tests conversion of FP4 values to float using E8M0 exponent scaling.
 *
 * This function performs a series of conversions from FP4 values to float values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 and FP4 values,
 * as well as specific vector and rounding conversions.
 *
 * @param N The maximum number of conversions to perform.
 * @param p_test Pointer to the output array where the converted float values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 * @note If either p_test or p_completed is nullptr, the function will return immediately.
 * @note The function will stop converting if the number of conversions reaches N.
 * @note First 256*16 conversions are for all possible combinations of E8M0 and FP4 values that are
 * stored in memory sequentially with FP4 values varying faster.
 *
 * The function performs the following conversions:
 * - All possible combinations of E8M0 and FP4 values. [256x16]
 * - Vector conversions f4x2 -> f32x2. [2]
 * - Vector conversions  f32x2 -> f4x2 rne. [2]
 * - Vector conversions  f32x2 -> f4x2 sr. [2]
 * - Round to nearest even conversions for specific float values. [6]
 *
 * The results are stored in the p_test array, and the number of completed conversions
 * is updated in the p_completed variable.
 */
__host__ __device__ void
test_mx_fp4_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
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

    // All possible combinations of E8M0 and FP4
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
        {
            uint8_t fp4_uid = static_cast<uint8_t>(fp4_id);
            auto v    = scaled_type_convert<float>(e8m0_bexp_t(exp_id), f4_t(fp4_uid & 0b00001111));
            p_test[i] = v;
            i++;
            if(i >= N)
            {
                return;
            }
        }
    }

    /// Test vector conversions
    // f4x2 -> f32x2
    f4x2_t f4x2{f4x2_t::data_v{0b00011100}}; // 0b0001(=0.5) and 0b1100(=-2.0)
    auto scale2 = e8m0_bexp_t(2.0f);

    float2_t f32x2 = scaled_type_convert<float2_t>(scale2, f4x2);
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

    // f32x2 -> f4x2
    f32x2 = {1.0f, -4.0f};
    f4x2  = f4_convert_rne(f32x2, type_convert<float>(scale2)); // expect {0.5, -2}

    p_test[i++] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<0>{}))); // 0.5f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<1>{}))); // -2.0f
    if(i >= N)
    {
        return;
    }

    f4x2 = f4_convert_sr(f32x2, type_convert<float>(scale2)); // expect {0.5, -2}

    p_test[i++] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<0>{}))); // 0.5f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(
        f4_t(f4x2.AsType<f4x2_pk_t>()(ck::Number<0>{}).unpack<>(ck::Number<1>{}))); // -2.0f
    if(i >= N)
    {
        return;
    }

    /// Test round to nearest even

    p_test[i++] = type_convert<float>(f4_convert_rne(24.0f, 4.0f)); // 24/4
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(
        f4_convert_rne(std::numeric_limits<float>::quiet_NaN(), 4.0f)); // => NaN
    if(i >= N)
    {
        return;
    }

    // Inf/2 > 6.0 => 6.0 on device
    p_test[i++] = type_convert<float>(f4_convert_rne(std::numeric_limits<float>::infinity(), 2.0f));
    if(i >= N)
    {
        return;
    }

    // 256/0.5  > 6.0 => 6.0 on device
    p_test[i++] = type_convert<float>(f4_convert_rne(256.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    // -256/0.5  < -6.0 => -6.0 on device
    p_test[i++] = type_convert<float>(f4_convert_rne(-256.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    // proper scale selection
    p_test[i++] = type_convert<float>(f4_convert_rne(20.0f, 4.0f)); // 20.0/4.0 = 5.0
    if(i >= N)
    {
        return;
    }
}

TEST(MXFP4, HostScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);
    uint64_t completed = 0;

    test_mx_fp4_scaled_convert(test_size, out.data(), &completed);

    // V = X * P; X - E8M0 scale, P - FP4

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
    {
        auto idx = e8m0_nan_id * 16 + fp4_id;
        ASSERT_TRUE(std::isnan(out[idx]));
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
        {
            uint8_t fp4_uid = static_cast<uint8_t>(fp4_id);
            auto idx        = exp_id * 16 + fp4_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(f4_t(fp4_uid & 0b00001111)))
                << "exp_id: " << exp_id << " fp4_id: " << fp4_id << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(f4_t(fp4_uid & 0b00001111));
        }
    }

    /// Test vector conversions

    auto i = 256 * 16;

    // f4x2 -> f32x2
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 1.0f);

    // f32x2 -> f4x2
    // RNE
    EXPECT_EQ(out[i++], 0.5f);
    EXPECT_EQ(out[i++], -2.0f);
    // SR
    EXPECT_EQ(out[i++], 0.5f);
    EXPECT_EQ(out[i++], -2.0f);

    /// Test round to nearest even
    EXPECT_EQ(out[i++], 24.0f / 4.0f) << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(type_convert<f4_t>(5.0f)))
        << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__global__ void test_mx_fp4_device_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
{
    test_mx_fp4_scaled_convert(N, p_test, p_completed);
}

TEST(MXFP4, DeviceScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp4_device_scaled_convert<<<1, 1>>>(
        test_size,
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    // V = X * P; X - E8M0 scale, P - FP4

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
    {
        auto idx = e8m0_nan_id * 16 + fp4_id;
        ASSERT_TRUE(std::isnan(out[idx])) << "idx: " << idx << " out[idx]: " << out[idx];
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t fp4_id = 0; fp4_id < 16; fp4_id++)
        {
            uint8_t fp4_uid = static_cast<uint8_t>(fp4_id);
            auto idx        = exp_id * 16 + fp4_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(f4_t(fp4_uid & 0b00001111)))
                << "exp_id: " << exp_id << " fp4_id: " << fp4_id << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(f4_t(fp4_uid & 0b00001111));
        }
    }

    /// Test vector conversions

    auto i = 256 * 16;

    // f4x2 -> f32x2
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 1.0f);

    // f32x2 -> f4x2
    // RNE
    EXPECT_EQ(out[i++], 0.5f);
    EXPECT_EQ(out[i++], -2.0f);
    // SR
    EXPECT_EQ(out[i++], 0.5f);
    EXPECT_EQ(out[i++], -2.0f);

    /// Test round to nearest even
    EXPECT_EQ(out[i++], 24.0f / 4.0f) << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<f4_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(type_convert<f4_t>(5.0f)))
        << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__host__ __device__ float vec16_generator(ck::index_t i, float scale)
{
    return scale * type_convert<float>(f4_t(i & 0b00001111));
}

__host__ __device__ float vec32_generator(ck::index_t i, float scale)
{
    if(i < 16)
    {
        return vec16_generator(
            i, scale); // all positive values, then all negative values in ascending order
    }
    else
    {
        return vec16_generator(
            15 - (i % 16),
            scale); // all negative values, then all positive values in descending order
    }
}

__global__ void test_mx_fp4x32_device_scaled_convert(float* p_test, uint64_t* p_completed)
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

    f4x32_t f4x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}([&](auto ii) {
        float32[static_cast<int>(ii)] = vec32_generator(ii, type_convert<float>(scale2));
    });

    f4x32 = f4_convert_rne(float32, type_convert<float>(scale2));

    ck::static_for<0, N / 2, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(
            f4_t(f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}).template unpack<>(ck::Number<0>{})));
        p_test[i++] = type_convert<float>(
            f4_t(f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}).template unpack<>(ck::Number<1>{})));
    });
}

TEST(MXFP4, DeviceF32x32ToF4x32ScaledConvert)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp4x32_device_scaled_convert<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i      = 0;
    auto scale2 = e8m0_bexp_t(2.0f);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++],
                  vec32_generator(ii, type_convert<float>(scale2)) / type_convert<float>(scale2))
            << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}

__global__ void test_mx_fp4x32_device_scaled_convert_sr(float* p_test, uint64_t* p_completed)
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

    f4x32_t f4x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}([&](auto ii) {
        float32[static_cast<int>(ii)] = vec32_generator(ii, type_convert<float>(scale2));
    });

    f4x32 = f4_convert_sr(float32, type_convert<float>(scale2));

    ck::static_for<0, N / 2, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(
            f4_t(f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}).template unpack<>(ck::Number<0>{})));
        p_test[i++] = type_convert<float>(
            f4_t(f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}).template unpack<>(ck::Number<1>{})));
    });
}

TEST(MXFP4, DeviceF32x32ToF4x32ScaledConvertSR)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_fp4x32_device_scaled_convert_sr<<<1, 1>>>(
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    auto i      = 0;
    auto scale2 = e8m0_bexp_t(2.0f);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++],
                  vec32_generator(ii, type_convert<float>(scale2)) / type_convert<float>(scale2))
            << "ii: " << ii << std::endl;
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

    auto scale2 = e8m0_bexp_t(2.0f);

    f4x32_t f4x32{};
    float32_t float32{};
    ck::static_for<0, N / 2, 1>{}([&](auto ii) {
        f4x32.AsType<f4x2_pk_t>()(ck::Number<ii>{}) = f4x2_pk_t{}.pack(
            type_convert<f4_t>(vec32_generator(2 * ii, type_convert<float>(scale2)) /
                               type_convert<float>(scale2)),
            type_convert<f4_t>(vec32_generator(2 * ii + 1, type_convert<float>(scale2)) /
                               type_convert<float>(scale2)));
    });

    float32 = scaled_type_convert<float32_t>(scale2, f4x32);

    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = float32[static_cast<int>(ii)]; });
}

TEST(MXFP4, DeviceF4x32ToF32x32ScaledConvert)
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

    auto i      = 0;
    auto scale2 = e8m0_bexp_t(2.0f);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        EXPECT_EQ(out[i++], vec32_generator(ii, type_convert<float>(scale2)))
            << "ii: " << ii << std::endl;
    });

    EXPECT_EQ(N, completed);
    EXPECT_EQ(N, i);
}
