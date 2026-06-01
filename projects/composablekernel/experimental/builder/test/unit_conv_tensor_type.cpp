// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"

namespace {

namespace ckb = ck_tile::builder;
using ck_tile::builder::factory::internal::DataTypeToCK;

template <ckb::DataType DT, typename T>
constexpr auto check_same = std::is_same_v<typename DataTypeToCK<DT>::type, T>;

TEST(ConvTensorType, Exhaustive)
{
    using enum ckb::DataType;

    const auto type = FP32;
    // This switch ensures that we get a warning (error with -Werror) if
    // a variant is missing.
    switch(type)
    {
    case UNDEFINED_DATA_TYPE: break;
    case FP32: EXPECT_TRUE((check_same<FP32, float>)); break;
    case FP16: EXPECT_TRUE((check_same<FP16, ck::half_t>)); break;
    case BF16: EXPECT_TRUE((check_same<BF16, ck::bhalf_t>)); break;
    case I32: EXPECT_TRUE((check_same<I32, uint32_t>)); break;
    case FP8: EXPECT_TRUE((check_same<FP8, ck::f8_t>)); break;
    case I8: EXPECT_TRUE((check_same<I8, int8_t>)); break;
    case U8: EXPECT_TRUE((check_same<U8, uint8_t>)); break;
    }
}

} // namespace
