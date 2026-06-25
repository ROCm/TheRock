// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_avg_pool2d_bwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename T>
class AvgPool2dBWDTest : public ::testing::Test
{
    protected:
    using InDataType  = std::tuple_element_t<0, T>;
    using OutDataType = std::tuple_element_t<1, T>;

    static std::vector<PoolingParam> params;

    void Run()
    {
        for(size_t i = 0; i < this->params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = this->params[i];
            bool success =
                ck::profiler::profile_avg_pool2d_bwd_impl<InDataType, OutDataType, NHWC, NHWC>(
                    true,
                    2,
                    false,
                    false,
                    param.length_,
                    param.window_spatial_lengths_,
                    param.window_strides_,
                    param.window_dilations_,
                    param.input_left_pads_,
                    param.input_right_pads_,
                    instance_index);
            EXPECT_TRUE(success);
        }
    }
};

template <typename T>
std::vector<PoolingParam> AvgPool2dBWDTest<T>::params = {
    {{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    {{1, 1, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
    {{1, 5, 7, 7}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, {0, 0}},
    {{1, 1, 8, 8}, {2, 2}, {2, 2}, {1, 1}, {2, 2}, {0, 0}},
    {{1, 1, 8, 8}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, {0, 0}},
    {{2, 32, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}},
    {{1, 2, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0}}};

using Avg_Pool_2D_f32_types  = ::testing::Types<std::tuple<F32, F32>>;
using Avg_Pool_2D_int8_types = ::testing::Types<std::tuple<I8, I8>>;
using Avg_Pool_2D_f16_types  = ::testing::Types<std::tuple<F16, F16>>;
using Avg_Pool_2D_bf16_types = ::testing::Types<std::tuple<BF16, BF16>>;
using Avg_Pool_2D_f8_types   = ::testing::Types<std::tuple<F8, F8>>;

template <typename TType>
class AvgPool2D_f32 : public AvgPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP32)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_f32 tests because CK_ENABLE_FP32 is not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_int8 : public AvgPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_INT8)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_int8 tests because CK_ENABLE_INT8 is not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_f16 : public AvgPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP16)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_f16 because CK_ENABLE_FP16 is not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_bf16 : public AvgPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_BF16)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_bf16 tests because CK_ENABLE_BF16 is not enabled";
        }
    }
};

template <typename TType>
class AvgPool2D_f8 : public AvgPool2dBWDTest<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP8)
        {
            GTEST_SKIP() << "Skipping AvgPool2D_f8 tests because CK_ENABLE_FP8 is not enabled";
        }
    }
};

TYPED_TEST_SUITE(AvgPool2D_f32, Avg_Pool_2D_f32_types);
TYPED_TEST_SUITE(AvgPool2D_int8, Avg_Pool_2D_int8_types);
TYPED_TEST_SUITE(AvgPool2D_f16, Avg_Pool_2D_f16_types);
TYPED_TEST_SUITE(AvgPool2D_bf16, Avg_Pool_2D_bf16_types);
TYPED_TEST_SUITE(AvgPool2D_f8, Avg_Pool_2D_f8_types);

TYPED_TEST(AvgPool2D_f32, AvgPool2DTest_f32) { this->Run(); }

TYPED_TEST(AvgPool2D_int8, AvgPool2DTest_int8) { this->Run(); }

TYPED_TEST(AvgPool2D_f16, AvgPool2DTest_f16) { this->Run(); }

TYPED_TEST(AvgPool2D_bf16, AvgPool2DTest_bf16) { this->Run(); }

TYPED_TEST(AvgPool2D_f8, AvgPool2DTest_f8) { this->Run(); }

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
