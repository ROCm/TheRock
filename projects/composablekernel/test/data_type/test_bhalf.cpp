// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include <hip/hip_runtime.h>

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"

using ::ck::hip_check_error;

using ck::bhalf_t;
using ck::type_convert;

TEST(BHALF_T, Nan)
{
    const uint16_t binary_bhalf_nan = 0x7FC0;
    const bhalf_t bhalf_nan         = ck::bit_cast<bhalf_t>(binary_bhalf_nan);
    EXPECT_EQ(bhalf_nan, type_convert<bhalf_t>(ck::NumericLimits<float>::QuietNaN()));
}

TEST(BHALF_T, Inf)
{
    const uint16_t binary_bhalf_inf = 0x7F80;
    const bhalf_t bhalf_inf         = ck::bit_cast<bhalf_t>(binary_bhalf_inf);
    EXPECT_EQ(bhalf_inf, type_convert<bhalf_t>(ck::NumericLimits<float>::Infinity()));
}

TEST(BHALF_T, MantisaOverflow)
{
    const float abs_tol   = std::pow(2, -7);
    const uint32_t val    = 0x81FFFFFF;
    const float float_val = ck::bit_cast<float>(val);

    ASSERT_NEAR(float_val, type_convert<float>(type_convert<bhalf_t>(float_val)), abs_tol);
}

TEST(BHALF_T, ExpOverflow)
{
    const uint32_t val    = 0xFF800000;
    const float float_val = ck::bit_cast<float>(val);
    ASSERT_EQ(type_convert<float>(type_convert<bhalf_t>(float_val)), float_val);
}

TEST(BHALF_T, MantisaExpOverflow)
{
    const uint32_t val    = 0xFFFFFFFF;
    const float float_val = ck::bit_cast<float>(val);

    ASSERT_TRUE(std::isnan(float_val));
    ASSERT_TRUE(std::isnan(type_convert<float>(type_convert<bhalf_t>(float_val))));
}

__global__ void cast(const float input, float* output)
{
    const bhalf_t bhalf_val = type_convert<bhalf_t>(input);
    *output                 = type_convert<float>(bhalf_val);
}

TEST(BHALF_T, CastOnDevice)
{
    constexpr int num_vals     = 11;
    const float abs_tol        = std::pow(2, -7);
    float float_vals[num_vals] = {0.5, 0.875, 1.5, 1, 2, 4, 8, 16, 32, 64, 128};

    float* float_val_after_cast_dev;
    float float_val_after_cast_host;
    hip_check_error(hipMalloc(&float_val_after_cast_dev, sizeof(float)));

    // Positive
    for(int idx = 0; idx < num_vals; idx++)
    {
        cast<<<1, 1>>>(float_vals[idx], float_val_after_cast_dev);

        hip_check_error(hipMemcpy(&float_val_after_cast_host,
                                  float_val_after_cast_dev,
                                  sizeof(float),
                                  hipMemcpyDeviceToHost));

        ASSERT_NEAR(float_val_after_cast_host, float_vals[idx], abs_tol);
    }
    // Negative
    for(int idx = 0; idx < num_vals; idx++)
    {
        cast<<<1, 1>>>(-float_vals[idx], float_val_after_cast_dev);

        hip_check_error(hipMemcpy(&float_val_after_cast_host,
                                  float_val_after_cast_dev,
                                  sizeof(float),
                                  hipMemcpyDeviceToHost));

        ASSERT_NEAR(float_val_after_cast_host, -float_vals[idx], abs_tol);
    }
}
