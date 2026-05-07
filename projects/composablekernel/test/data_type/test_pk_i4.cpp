// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <bitset>
#include <cinttypes>
#include <cstdint>
#include <iomanip>
#include "gtest/gtest.h"
#include <hip/hip_runtime.h>

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/library/utility/device_memory.hpp"

#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

using ::ck::DeviceMem;

using ck::bhalf2_t;
using ck::bhalf_t;
using ck::float2_t;
using ck::half2_t;
using ck::half4_t;
using ck::half_t;
using ck::pk_i4_t;
using ck::pk_i4x4_t;

TEST(PackedInt4, ConvertToFloat)
{
#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    constexpr float first_input_val  = 7.f;
    constexpr float second_input_val = -1.f;
#else
    constexpr float first_input_val  = -1.f;
    constexpr float second_input_val = 7.f;
#endif
    uint8_t data = 0b11110111; // {-1, 7}
    pk_i4_t in   = ck::bit_cast<int8_t>(data);
    float2_t out = ck::type_convert<float2_t>(in);

    EXPECT_EQ(out.x, first_input_val);
    EXPECT_EQ(out.y, second_input_val);
}

TEST(PackedInt4, ConvertToHalf)
{
#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    constexpr half_t first_input_val  = ck::type_convert<half_t>(7.f);
    constexpr half_t second_input_val = ck::type_convert<half_t>(-1.f);
#else
    constexpr half_t first_input_val  = ck::type_convert<half_t>(-1.f);
    constexpr half_t second_input_val = ck::type_convert<half_t>(7.f);
#endif
    uint8_t data = 0b11110111; // {-1, 7}
    pk_i4_t in   = ck::bit_cast<int8_t>(data);
    half2_t out  = ck::type_convert<half2_t>(in);

    EXPECT_EQ(out.x, first_input_val);
    EXPECT_EQ(out.y, second_input_val);
}

TEST(PackedInt4, ConvertToBHalf)
{
#ifdef CK_USE_PK4_LAYOUT_SHUFFLE
    const bhalf_t first_input_val  = ck::type_convert<bhalf_t>(7.f);
    const bhalf_t second_input_val = ck::type_convert<bhalf_t>(-1.f);
#else
    const bhalf_t first_input_val  = ck::type_convert<bhalf_t>(-1.f);
    const bhalf_t second_input_val = ck::type_convert<bhalf_t>(7.f);
#endif
    uint8_t data = 0b11110111; // {-1, 7}
    pk_i4_t in   = ck::bit_cast<int8_t>(data);
    bhalf2_t out = ck::type_convert<bhalf2_t>(in);

    EXPECT_EQ(out.x, first_input_val);
    EXPECT_EQ(out.y, second_input_val);
}
