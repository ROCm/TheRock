// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_conv_tensor_rearrange_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestConvTensorRearrange : public ::testing::Test
{
    protected:
    using ImLayout              = std::tuple_element_t<0, Tuple>;
    using ConvTensorRearrangeOp = std::tuple_element_t<1, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial, typename InDataType, typename OutDataType>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(size_t i = 0; i < conv_params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = conv_params[i];
            pass        = pass && ck::profiler::profile_conv_tensor_rearrange_impl<NDimSpatial,
                                                                                   ImLayout,
                                                                                   InDataType,
                                                                                   OutDataType,
                                                                                   ConvTensorRearrangeOp>(
                               true,  // do_verification
                               1,     // init_method: integer value
                               false, // do_log
                               false, // time_kernel
                               param,
                               instance_index);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;
using namespace ck::conv_tensor_rearrange_op;

using KernelTypes1d = ::testing::Types<std::tuple<GNWC, ImageToColumn>,
                                       std::tuple<GNWC, ColumnToImage>,
                                       std::tuple<NWGC, ImageToColumn>,
                                       std::tuple<NWGC, ColumnToImage>>;

using KernelTypes2d = ::testing::Types<std::tuple<GNHWC, ImageToColumn>,
                                       std::tuple<GNHWC, ColumnToImage>,
                                       std::tuple<NHWGC, ImageToColumn>,
                                       std::tuple<NHWGC, ColumnToImage>>;

using KernelTypes3d = ::testing::Types<std::tuple<GNDHWC, ImageToColumn>,
                                       std::tuple<GNDHWC, ColumnToImage>,
                                       std::tuple<NDHWGC, ImageToColumn>,
                                       std::tuple<NDHWGC, ColumnToImage>>;

template <typename Tuple>
class TestConvTensorRearrange1d : public TestConvTensorRearrange<Tuple>
{
};

template <typename Tuple>
class TestConvTensorRearrange2d : public TestConvTensorRearrange<Tuple>
{
};

template <typename Tuple>
class TestConvTensorRearrange3d : public TestConvTensorRearrange<Tuple>
{
};

TYPED_TEST_SUITE(TestConvTensorRearrange1d, KernelTypes1d);
TYPED_TEST_SUITE(TestConvTensorRearrange2d, KernelTypes2d);
TYPED_TEST_SUITE(TestConvTensorRearrange3d, KernelTypes3d);

TYPED_TEST(TestConvTensorRearrange1d, Test1D)
{
    this->conv_params.clear();

    this->conv_params.push_back({1, 2, 4, 1, 192, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 2, 64, 1, 64, {3}, {14}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 2, 64, 1, 64, {1}, {7}, {3}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 2, 64, 1, 64, {1}, {3}, {1}, {1}, {0}, {0}});
    // ScalarPerVector should be 1
    this->conv_params.push_back({1, 2, 4, 1, 1, {3}, {28}, {1}, {1}, {1}, {1}});
    // stride != 1
    this->conv_params.push_back({1, 2, 1, 1, 4, {3}, {28}, {2}, {1}, {1}, {1}});
    // dilation != 1
    this->conv_params.push_back({1, 2, 1, 1, 4, {3}, {28}, {1}, {2}, {1}, {1}});
#ifdef CK_ENABLE_FP32
    this->template Run<1, float, float>();
#endif
#ifdef CK_ENABLE_BF16
    this->template Run<1, ck::bhalf_t, ck::bhalf_t>();
#endif
#ifdef CK_ENABLE_FP16
    this->template Run<1, ck::half_t, ck::half_t>();
#endif
#ifdef CK_ENABLE_INT8
    this->template Run<1, int8_t, int8_t>();
#endif
}

TYPED_TEST(TestConvTensorRearrange2d, Test2D)
{
    this->conv_params.clear();

    this->conv_params.push_back(
        {2, 2, 4, 1, 192, {3, 3}, {28, 28}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 2, 64, 1, 64, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 64, 1, 64, {1, 1}, {7, 7}, {3, 3}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 64, 1, 64, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 64, 1, 64, {3, 3}, {28, 28}, {2, 2}, {2, 2}, {1, 1}, {1, 1}});
#ifdef CK_ENABLE_FP32
    this->template Run<2, float, float>();
#endif
#ifdef CK_ENABLE_BF16
    this->template Run<2, ck::bhalf_t, ck::bhalf_t>();
#endif
#ifdef CK_ENABLE_FP16
    this->template Run<2, ck::half_t, ck::half_t>();
#endif
#ifdef CK_ENABLE_INT8
    this->template Run<2, int8_t, int8_t>();
#endif
}

TYPED_TEST(TestConvTensorRearrange3d, Test3D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {3, 2, 16, 1, 64, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {3, 3, 3}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 2, 1, 64, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 2, 32, 1, 64, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 64, 1, 64, {3, 3, 3}, {14, 14, 14}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}});
#ifdef CK_ENABLE_FP32
    this->template Run<3, float, float>();
#endif
#ifdef CK_ENABLE_BF16
    this->template Run<3, ck::bhalf_t, ck::bhalf_t>();
#endif
#ifdef CK_ENABLE_FP16
    this->template Run<3, ck::half_t, ck::half_t>();
#endif
#ifdef CK_ENABLE_INT8
    this->template Run<3, int8_t, int8_t>();
#endif
}
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
