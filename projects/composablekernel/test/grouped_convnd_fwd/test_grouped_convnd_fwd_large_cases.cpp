// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_fwd_impl.hpp"

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
        for(auto& param : conv_params)
        {
            pass = pass && ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                       InLayout,
                                                                       WeiLayout,
                                                                       OutLayout,
                                                                       DataType,
                                                                       DataType,
                                                                       DataType,
                                                                       DataType,
                                                                       DataType,
                                                                       IndexType>(
                               true,  // do_verification
                               1,     // init_method: integer value
                               false, // do_log
                               false, // time_kernel
                               param);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes2d = ::testing::Types<std::tuple<float, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<ck::half_t, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<ck::bhalf_t, NHWGC, GKYXC, NHWGK>,
                                       std::tuple<int8_t, NHWGC, GKYXC, NHWGK>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<ck::half_t, NDHWGC, GKZYXC, NDHWGK>,
                                       std::tuple<ck::bhalf_t, NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwd2d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd3d : public TestGroupedConvndFwd<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwd2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndFwd3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwd2d, Test2D)
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

TYPED_TEST(TestGroupedConvndFwd3d, Test3D)
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
