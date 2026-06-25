// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_fwd_bias_clamp_impl.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
static ck::index_t param_mask     = 0xffffff;
static ck::index_t instance_index = -1;
using AddClamp                    = ck::tensor_operation::element_wise::AddClamp;

template <typename Tuple>
class TestGroupedConvndFwd : public ::testing::Test
{
    protected:
    using DataType  = std::tuple_element_t<0, Tuple>;
    using InLayout  = std::tuple_element_t<1, Tuple>;
    using WeiLayout = std::tuple_element_t<2, Tuple>;
    using OutLayout = std::tuple_element_t<3, Tuple>;
    using IndexType = ck::long_index_t;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
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
            pass = pass && ck::profiler::profile_grouped_conv_fwd_bias_clamp_impl<NDimSpatial,
                                                                                  InLayout,
                                                                                  WeiLayout,
                                                                                  OutLayout,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  DataType,
                                                                                  IndexType,
                                                                                  false /*BiasGK*/>(
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

using KernelTypes2d = ::testing::Types<std::tuple<ck::bhalf_t, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<float, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<ck::half_t, NHWGC, GKYXC, NHWGK>>;

using KernelTypes3d = ::testing::Types<std::tuple<ck::bhalf_t, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<float, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<ck::half_t, NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwdBiasClamp2d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwdBiasClamp3d : public TestGroupedConvndFwd<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwdBiasClamp2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndFwdBiasClamp3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwdBiasClamp2d, Test2D)
{
    // Case larger than 2GB
    this->conv_params.push_back(
        {2, 1, 128, 4, 192, {2, 2}, {224, 224}, {224, 224}, {1, 1}, {0, 0}, {0, 0}});
    // With supported NumGroupsToMerge > 1
    this->conv_params.push_back(
        {2, 32, 64, 1, 1, {2, 2}, {672, 672}, {672, 672}, {1, 1}, {0, 0}, {0, 0}});
    // When image is larger than 2GB
    this->conv_params.push_back(
        {2, 2, 2, 128, 128, {3, 3}, {4096, 2048}, {300, 300}, {3, 3}, {1, 1}, {1, 1}});
    // Split N and G > 1
    this->conv_params.push_back(
        {2, 4, 112, 8, 8, {3, 3}, {469, 724}, {2, 2}, {2, 2}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndFwdBiasClamp3d, Test3D)
{
    // Case larger than 2GB
    this->conv_params.push_back({3,
                                 1,
                                 128,
                                 4,
                                 192,
                                 {2, 2, 2},
                                 {2, 224, 224},
                                 {1, 224, 224},
                                 {1, 1, 1},
                                 {0, 0, 0},
                                 {0, 0, 0}});
    // With supported NumGroupsToMerge > 1
    this->conv_params.push_back({3,
                                 32,
                                 64,
                                 1,
                                 1,
                                 {2, 2, 2},
                                 {360, 2, 672},
                                 {360, 2, 672},
                                 {1, 1, 1},
                                 {0, 0, 0},
                                 {0, 0, 0}});
    // When image is larger than 2GB
    this->conv_params.push_back({3,
                                 1,
                                 2,
                                 128,
                                 128,
                                 {3, 1, 3},
                                 {900, 2, 2048},
                                 {300, 1, 300},
                                 {3, 2, 3},
                                 {1, 1, 1},
                                 {1, 1, 1}});
    this->template Run<3>();
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
