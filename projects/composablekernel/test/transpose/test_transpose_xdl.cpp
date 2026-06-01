// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_transpose_impl.hpp"

using F16 = ck::half_t;
using F32 = float;
using ck::index_t;

template <typename Tuple>
class TestTranspose : public ::testing::Test
{
    protected:
    using ADataType = std::tuple_element_t<0, Tuple>;
    using BDataType = std::tuple_element_t<1, Tuple>;

    void Run()
    {
        std::vector<std::vector<ck::index_t>> lengths = {
            {4, 16, 16, 32, 5}, {8, 16, 16, 32, 8} /**{32, 16, 16, 32, 8},**/};

        for(auto length : lengths)
        {
            bool success = ck::profiler::profile_transpose_impl<ADataType, BDataType, 5>(
                true, 2, false, false, length);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16>, std::tuple<F32, F32>>;

TYPED_TEST_SUITE(TestTranspose, KernelTypes);
TYPED_TEST(TestTranspose, Test_FP16) { this->Run(); }
TYPED_TEST(TestTranspose, Test_FP32) { this->Run(); }
