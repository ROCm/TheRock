// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_permute_scale_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestPermute : public ::testing::Test
{
    protected:
    using ADataType = std::tuple_element_t<0, Tuple>;
    using BDataType = std::tuple_element_t<1, Tuple>;

    constexpr bool skip_case()
    {
#ifndef CK_ENABLE_FP16
        if constexpr(ck::is_same_v<ADataType, F16> || ck::is_same_v<BDataType, F16>)
        {
            return true;
        }
#endif
#ifndef CK_ENABLE_FP32
        if constexpr(ck::is_same_v<ADataType, F32> || ck::is_same_v<BDataType, F32>)
        {
            return true;
        }
#endif
        return false;
    }

    template <ck::index_t NDims>
    void Run(std::vector<ck::index_t> lengths,
             std::vector<ck::index_t> input_strides,
             std::vector<ck::index_t> output_strides)
    {
        if(!skip_case())
        {
            bool success = ck::profiler::profile_permute_scale_impl<ADataType, BDataType, NDims>(
                true, 2, false, false, lengths, input_strides, output_strides);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16>, std::tuple<F32, F32>>;

TYPED_TEST_SUITE(TestPermute, KernelTypes);
TYPED_TEST(TestPermute, Test1D)
{
    constexpr ck::index_t NumDims = 1;
    this->template Run<NumDims>({16}, {1}, {1});
    this->template Run<NumDims>({16}, {1}, {2});
    this->template Run<NumDims>({1}, {1}, {1});
}

TYPED_TEST(TestPermute, Test2D)
{
    constexpr ck::index_t NumDims = 2;
    this->template Run<NumDims>({8, 16}, {16, 1}, {1, 8});
    this->template Run<NumDims>({8, 16}, {1, 8}, {16, 1});
    this->template Run<NumDims>({1, 1}, {1, 1}, {1, 1});
}

TYPED_TEST(TestPermute, Test3D)
{
    constexpr ck::index_t NumDims = 3;
    this->template Run<NumDims>({8, 2, 8}, {16, 8, 1}, {1, 8, 16});
    this->template Run<NumDims>({8, 2, 8}, {1, 8, 16}, {16, 8, 1});
    this->template Run<NumDims>({1, 1, 1}, {1, 1, 1}, {1, 1, 1});
}

TYPED_TEST(TestPermute, Test4D)
{
    constexpr ck::index_t NumDims = 4;
    this->template Run<NumDims>({8, 2, 3, 8}, {48, 24, 8, 1}, {1, 8, 16, 48});
    this->template Run<NumDims>({8, 2, 3, 8}, {1, 8, 16, 48}, {48, 24, 8, 1});
    this->template Run<NumDims>({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
}

TYPED_TEST(TestPermute, Test5D)
{
    constexpr ck::index_t NumDims = 5;
    this->template Run<NumDims>({8, 2, 3, 4, 8}, {192, 96, 32, 8, 1}, {1, 8, 16, 48, 192});
    this->template Run<NumDims>({8, 2, 3, 4, 8}, {1, 8, 16, 48, 192}, {192, 96, 32, 8, 1});
    this->template Run<NumDims>({1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1});
}

TYPED_TEST(TestPermute, Test6D)
{
    constexpr ck::index_t NumDims = 6;
    this->template Run<NumDims>(
        {8, 2, 3, 4, 5, 8}, {960, 480, 160, 40, 8, 1}, {1, 8, 16, 48, 192, 960});
    this->template Run<NumDims>(
        {8, 2, 3, 4, 5, 8}, {1, 8, 16, 48, 192, 960}, {960, 480, 160, 40, 8, 1});
    this->template Run<NumDims>({1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1});
}
