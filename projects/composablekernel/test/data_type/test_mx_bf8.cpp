// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/library/utility/device_memory.hpp"
#include "ck/utility/scaled_type_convert.hpp"

using ::ck::DeviceMem;

using ck::bf8_ocp_t;
using ck::bf8x16_ocp_t;
using ck::bf8x2_ocp_t;
using ck::bf8x32_ocp_t;
using ck::e8m0_bexp_t;
using ck::float16_t;
using ck::float2_t;
using ck::float32_t;
using ck::mxf8_convert_rne;
using ck::mxf8_convert_sr;
using ck::scaled_type_convert;
using ck::type_convert;

constexpr uint64_t test_size = 256 * 256 + 2 + 4 + 6;

/**
 * @brief  Tests conversion of BF8 values to float using E8M0 exponent scaling.
 *
 * This function performs a series of conversions from BF8 values to float values using
 * E8M0 exponent scaling. It handles all possible combinations of E8M0 and BF8 values,
 * as well as specific vector and rounding conversions.
 *
 * @param N The maximum number of conversions to perform.
 * @param p_test Pointer to the output array where the converted float values will be stored.
 * @param p_completed Pointer to a variable that tracks the number of completed conversions.
 *
 * @note If either p_test or p_completed is nullptr, the function will return immediately.
 * @note The function will stop converting if the number of conversions reaches N.
 * @note First 256*256 conversions are for all possible combinations of E8M0 and BF8 values that are
 * stored in memory sequentially with BF8 values varying faster.
 *
 * The function performs the following conversions:
 * - All possible combinations of E8M0 and BF8 values. [256x256]
 * - Vector conversions bf8x2 -> f32x2. [2]
 * - Vector conversions  f32x2 -> bf8x2 rne. [2]
 * - Vector conversions  f32x2 -> bf8x2 sr. [2]
 * - Round to nearest even conversions for specific float values. [6]
 *
 * The results are stored in the p_test array, and the number of completed conversions
 * is updated in the p_completed variable.
 */
__host__ __device__ void
test_mx_bf8_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
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

    // All possible combinations of E8M0 and BF8
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
        {
            uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
            auto v          = scaled_type_convert<float>(e8m0_bexp_t(exp_id), bf8_ocp_t{bf8_uid});
            p_test[i]       = v;
            i++;
            if(i >= N)
            {
                return;
            }
        }
    }

    /// Test vector conversions
    // bf8x2 -> f32x2
    bf8x2_ocp_t bf8x2{bf8x2_ocp_t::data_v{0b10000100, 0b00000001}}; //-2^-14, 2^-16
    auto scale = e8m0_bexp_t(8.0f);

    float2_t f32x2 = scaled_type_convert<float2_t>(scale, bf8x2);
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

    // f32x2 -> bf8x2
    f32x2       = {-8.0f, 4.0f};
    auto scale2 = e8m0_bexp_t(2.0f);

    bf8x2 = mxf8_convert_rne<bf8x2_ocp_t>(f32x2, type_convert<float>(scale2)); // expect {-4, 2}

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

    auto scale4 = e8m0_bexp_t(4.0f);

    bf8x2 = mxf8_convert_sr<bf8x2_ocp_t>(f32x2, type_convert<float>(scale4)); // expect {-2, 1}

    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<0>{})); //-2f
    if(i >= N)
    {
        return;
    }
    p_test[i++] = type_convert<float>(bf8x2.AsType<bf8_ocp_t>()(ck::Number<1>{})); // 1f
    if(i >= N)
    {
        return;
    }

    /// Test round to nearest even

    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(1024.0f, 4.0f)); // 1024/4
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(
        mxf8_convert_rne<bf8_ocp_t>(std::numeric_limits<float>::quiet_NaN(), 4.0f)); // => NaN
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(
        std::numeric_limits<float>::infinity(), 2.0f)); // => BF8 Inf on device
    if(i >= N)
    {
        return;
    }

    // 31000/0.5 > 57344 => BF8 Inf on device
    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(31000.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    // -31000/0.5 < -57344  => -BF8 Inf on device
    p_test[i++] = type_convert<float>(mxf8_convert_rne<bf8_ocp_t>(-31000.0f, 0.5f));
    if(i >= N)
    {
        return;
    }

    p_test[i++] = type_convert<float>(
        mxf8_convert_rne<bf8_ocp_t>(powf(2.0f, 16.0f), 4.0f)); // 2^16/4 = 65536/4
    if(i >= N)
    {
        return;
    }
}

TEST(MXBF8, HostScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);
    uint64_t completed = 0;

    test_mx_bf8_scaled_convert(test_size, out.data(), &completed);

    // V = X * P; X - E8M0 scale, P - BF8

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        auto idx = e8m0_nan_id * 256 + bf8_id;
        ASSERT_TRUE(std::isnan(out[idx]));
    }

    // If P in {Inf, NaN}, then V = P
    std::set<uint8_t> bf8_spec_ids;
    bf8_spec_ids.insert(0b11111111); // -NaN
    bf8_spec_ids.insert(0b01111111); // +NaN
    bf8_spec_ids.insert(0b11111101); // -NaN
    bf8_spec_ids.insert(0b01111101); // +NaN
    bf8_spec_ids.insert(0b11111110); // -NaN
    bf8_spec_ids.insert(0b01111110); // +NaN
    bf8_spec_ids.insert(0b11111100); // -inf
    bf8_spec_ids.insert(0b01111100); // +inf
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(auto bf8_spec_id : bf8_spec_ids)
        {
            auto idx = exp_id * 256 + bf8_spec_id;

            if(std::isnan(type_convert<float>(bf8_ocp_t{bf8_spec_id})))
            {
                ASSERT_TRUE(std::isnan(out[idx]))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
            else
            {
                ASSERT_EQ(out[idx], type_convert<float>(bf8_ocp_t{bf8_spec_id}))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
        }
    }

    // V = X * P; X, P - finite
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
        {
            if(bf8_spec_ids.find(bf8_id) != bf8_spec_ids.end())
                continue;

            uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
            auto idx        = exp_id * 256 + bf8_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(bf8_ocp_t{bf8_uid}))
                << "exp_id: " << exp_id << " bf8_id: " << bf8_uid << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(bf8_ocp_t{bf8_uid});
        }
    }

    /// Test vector conversions

    auto i = 256 * 256;

    // bf8x2 -> f32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -11.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -13.0f));

    // f32x2 -> bf8x2
    // RNE
    EXPECT_EQ(out[i++], -4.0f);
    EXPECT_EQ(out[i++], 2.0f);
    // SR
    EXPECT_EQ(out[i++], -2.0f);
    EXPECT_EQ(out[i++], 1.0f);

    /// Test round to nearest even
    EXPECT_EQ(out[i++], 1024.0f / 4.0f) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isnan(out[i++])) << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], powf(2.0f, 14.0f)) << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__global__ void test_mx_bf8_device_scaled_convert(uint64_t N, float* p_test, uint64_t* p_completed)
{
    test_mx_bf8_scaled_convert(N, p_test, p_completed);
}

TEST(MXBF8, DeviceScaledConvert)
{
    std::vector<float> out(test_size, -1.0f);

    DeviceMem device_out(test_size * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8_device_scaled_convert<<<1, 1>>>(
        test_size,
        static_cast<float*>(device_out.GetDeviceBuffer()),
        static_cast<uint64_t*>(device_completed.GetDeviceBuffer()));

    uint64_t completed = 0;
    device_completed.FromDevice(&completed);
    device_out.FromDevice(out.data());

    // V = X * P; X - E8M0 scale, P - BF8

    // If X = NaN, then V = NaN regardless of P
    uint8_t e8m0_nan_id = ck::NumericLimits<e8m0_bexp_t>::QuietNaN().data;
    for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
    {
        auto idx = e8m0_nan_id * 256 + bf8_id;
        ASSERT_TRUE(std::isnan(out[idx])) << "idx: " << idx << " out[idx]: " << out[idx];
    }

    // If P in {Inf, NaN}, then V = P
    std::set<uint8_t> bf8_spec_ids;
    bf8_spec_ids.insert(0b11111111); //-NaN
    bf8_spec_ids.insert(0b01111111); // +NaN
    bf8_spec_ids.insert(0b11111101); //-NaN
    bf8_spec_ids.insert(0b01111101); // +NaN
    bf8_spec_ids.insert(0b11111110); //-NaN
    bf8_spec_ids.insert(0b01111110); // +NaN
    bf8_spec_ids.insert(0b11111100); //-inf
    bf8_spec_ids.insert(0b01111100); // +inf
    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(auto bf8_spec_id : bf8_spec_ids)
        {
            auto idx = exp_id * 256 + bf8_spec_id;

            if(std::isnan(type_convert<float>(bf8_ocp_t{bf8_spec_id})))
            {
                ASSERT_TRUE(std::isnan(out[idx]))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
            else
            {
                ASSERT_EQ(out[idx], type_convert<float>(bf8_ocp_t{bf8_spec_id}))
                    << "exp_id: " << exp_id << " bf8_id: " << bf8_spec_id << std::endl
                    << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                    << type_convert<float>(bf8_ocp_t{bf8_spec_id}) << " != " << out[idx];
            }
        }
    }

    for(ck::index_t exp_id = 0; exp_id < 256; exp_id++)
    {
        if(exp_id == e8m0_nan_id)
            continue;
        for(ck::index_t bf8_id = 0; bf8_id < 256; bf8_id++)
        {
            if(bf8_spec_ids.find(bf8_id) != bf8_spec_ids.end())
                continue;

            uint8_t bf8_uid = static_cast<uint8_t>(bf8_id);
            auto idx        = exp_id * 256 + bf8_uid;
            ASSERT_FLOAT_EQ(out[idx],
                            type_convert<float>(e8m0_bexp_t(exp_id)) *
                                type_convert<float>(bf8_ocp_t{bf8_uid}))
                << "exp_id: " << exp_id << " bf8_id: " << bf8_uid << std::endl
                << type_convert<float>(e8m0_bexp_t(exp_id)) << " * "
                << type_convert<float>(bf8_ocp_t{bf8_uid});
        }
    }

    /// Test vector conversions

    auto i = 256 * 256;

    // bf8x2 -> f32x2
    EXPECT_EQ(out[i++], -powf(2.0f, -11.0f));
    EXPECT_EQ(out[i++], powf(2.0f, -13.0f));

    // f32x2 -> bf8x2
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
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
    EXPECT_TRUE(std::isinf(out[i++])) << "out[i-1]: " << out[i - 1];
#else
    // NOTE: Host and Device have different behavior.
    // Device returns Infs, while Host returns Max (saturation to finite value).
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Max()))
        << "out[i-1]: " << out[i - 1];
    EXPECT_EQ(out[i++], type_convert<float>(ck::NumericLimits<bf8_ocp_t>::Lowest()))
        << "out[i-1]: " << out[i - 1];
#endif
    EXPECT_EQ(out[i++], powf(2.0f, 14.0f)) << "out[i-1]: " << out[i - 1];

    EXPECT_EQ(test_size, completed);
    EXPECT_EQ(test_size, i);
}

__host__ __device__ float vec16_generator(ck::index_t i) { return powf(-1.0f, i) * powf(2.0f, i); }

__global__ void test_mx_bf8x16_device_scaled_convert(float* p_test, uint64_t* p_completed)
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

    bf8x16_ocp_t bf8x16{};
    float16_t float16{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float16[static_cast<int>(ii)] = vec16_generator(ii); });

    bf8x16 = scaled_type_convert<bf8x16_ocp_t>(scale2, float16);

    ck::static_for<0, N, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(bf8x16.AsType<bf8_ocp_t>()(ck::Number<ii>{}));
    });
}

TEST(MXBF8, DeviceF32x16ToBF8x16ScaledConvert)
{
    constexpr int N = 16;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8x16_device_scaled_convert<<<1, 1>>>(
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

__global__ void test_mx_bf8x32_device_scaled_convert(float* p_test, uint64_t* p_completed)
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

    bf8x32_ocp_t bf8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float32[static_cast<int>(ii)] = vec32_generator(ii); });

    bf8x32 = mxf8_convert_rne<bf8x32_ocp_t>(float32, type_convert<float>(scale2));

    ck::static_for<0, N, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(bf8x32.AsType<bf8_ocp_t>()(ck::Number<ii>{}));
    });
}

TEST(MXBF8, DeviceF32x32ToBF8x32ScaledConvert)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8x32_device_scaled_convert<<<1, 1>>>(
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

__global__ void test_mx_bf8x32_device_scaled_convert_sr(float* p_test, uint64_t* p_completed)
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

    bf8x32_ocp_t bf8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}(
        [&](auto ii) { float32[static_cast<int>(ii)] = vec32_generator(ii); });

    bf8x32 = mxf8_convert_sr<bf8x32_ocp_t>(float32, type_convert<float>(scale2));

    ck::static_for<0, N, 1>{}([&](auto ii) {
        p_test[i++] = type_convert<float>(bf8x32.AsType<bf8_ocp_t>()(ck::Number<ii>{}));
    });
}

TEST(MXBF8, DeviceF32x32ToBF8x32ScaledConvertSR)
{
    constexpr int N = 32;
    std::vector<float> out(N, -1.0f);

    DeviceMem device_out(N * sizeof(float));
    DeviceMem device_completed(sizeof(uint64_t));

    device_out.SetValue(-21.0f);
    device_completed.SetValue(-21.0f);

    test_mx_bf8x32_device_scaled_convert_sr<<<1, 1>>>(
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

    bf8x32_ocp_t bf8x32{};
    float32_t float32{};
    ck::static_for<0, N, 1>{}([&](auto ii) {
        bf8x32.AsType<bf8_ocp_t>()(ii) = type_convert<bf8_ocp_t>(vec32_generator(ii) / 16.0f);
    });

    float32 = scaled_type_convert<float32_t>(scale2, bf8x32);

    ck::static_for<0, N, 1>{}([&](auto ii) { p_test[i++] = float32[static_cast<int>(ii)]; });
}

TEST(MXBF8, DeviceBF8x32ToF32x32ScaledConvert)
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
