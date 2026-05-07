// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "profiler/profile_pool2d_fwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestMaxPool2dFwd : public ::testing::Test
{
    protected:
    using InDataType                  = std::tuple_element_t<0, Tuple>;
    using OutDataType                 = std::tuple_element_t<1, Tuple>;
    using ComputeDataType             = std::tuple_element_t<2, Tuple>;
    using IndexDataType               = std::tuple_element_t<3, Tuple>;
    static constexpr bool ReturnIndex = std::tuple_element_t<4, Tuple>::value;

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
            // max pool
            bool success =
                ck::profiler::profile_pool2d_fwd_impl<InDataType,
                                                      OutDataType,
                                                      ComputeDataType,
                                                      IndexDataType,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::tensor_layout::convolution::NHWC,
                                                      ck::ReduceTensorOp::MAX,
                                                      false,
                                                      ReturnIndex>(true,
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
std::vector<PoolingParam> TestMaxPool2dFwd<T>::params = {
    {{{1, 1, 1, 1}, {1, 1}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
     {{2, 16, 64, 64}, {64, 64}, {1, 1}, {1, 1}, {0, 0}, {0, 0}},
     {{2, 16, 64, 64}, {4, 4}, {4, 4}, {2, 2}, {0, 0}, {0, 0}},
     {{2, 32, 30, 30}, {2, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1}}}};

using true_t  = std::integral_constant<bool, true>;
using false_t = std::integral_constant<bool, false>;

using MaxPool2D_F32_Types  = ::testing::Types<std::tuple<F32, F32, F32, I32, true_t>,
                                              std::tuple<F32, F32, F32, I32, false_t>>;
using MaxPool2D_F16_Types  = ::testing::Types<std::tuple<F16, F16, F32, I32, true_t>,
                                              std::tuple<F16, F16, F32, I32, false_t>>;
using MaxPool2D_BF16_Types = ::testing::Types<std::tuple<I8, I8, F32, I32, true_t>,
                                              std::tuple<BF16, BF16, F32, I32, false_t>>;
using MaxPool2D_I8_Types =
    ::testing::Types<std::tuple<I8, I8, F32, I32, true_t>, std::tuple<I8, I8, F32, I32, false_t>>;
using MaxPool2D_F8_Types =
    ::testing::Types<std::tuple<F8, F8, F32, I32, true_t>, std::tuple<F8, F8, F32, I32, false_t>>;

template <typename TType>
class MaxPool2D_F32 : public TestMaxPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP32)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_F32 tests because CK_ENABLE_FP32 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class MaxPool2D_F16 : public TestMaxPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP16)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_F16 tests because CK_ENABLE_FP16 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class MaxPool2D_BF16 : public TestMaxPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_BF16)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_BF16 tests because CK_ENABLE_BF16 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class MaxPool2D_I8 : public TestMaxPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_INT8)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_I8 tests because CK_ENABLE_INT8 is "
                            "not enabled";
        }
    }
};

template <typename TType>
class MaxPool2D_F8 : public TestMaxPool2dFwd<TType>
{
    protected:
    void SetUp() override
    {
        if(!CK_ENABLE_FP8)
        {
            GTEST_SKIP() << "Skipping MaxPool2D_F8 tests because CK_ENABLE_FP8 is "
                            "not enabled";
        }
    }
};

TYPED_TEST_SUITE(MaxPool2D_F32, MaxPool2D_F32_Types);
TYPED_TEST_SUITE(MaxPool2D_F16, MaxPool2D_F16_Types);
TYPED_TEST_SUITE(MaxPool2D_BF16, MaxPool2D_BF16_Types);
TYPED_TEST_SUITE(MaxPool2D_I8, MaxPool2D_I8_Types);
TYPED_TEST_SUITE(MaxPool2D_F8, MaxPool2D_F8_Types);

TYPED_TEST(MaxPool2D_F32, MaxPool2D_F32_Test) { this->Run(); }
TYPED_TEST(MaxPool2D_F16, MaxPool2D_F16_Test) { this->Run(); }
TYPED_TEST(MaxPool2D_BF16, MaxPool2D_BF16_Test) { this->Run(); }
TYPED_TEST(MaxPool2D_I8, MaxPool2D_I8_Test) { this->Run(); }
TYPED_TEST(MaxPool2D_F8, MaxPool2D_F8_Test) { this->Run(); }

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
