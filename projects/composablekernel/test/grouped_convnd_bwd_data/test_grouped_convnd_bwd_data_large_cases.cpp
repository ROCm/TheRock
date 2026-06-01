// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "profiler/profile_grouped_conv_bwd_data_impl.hpp"

template <typename Tuple>
class TestGroupedConvndBwdData : public ::testing::Test
{
    protected:
    using DataType  = std::tuple_element_t<0, Tuple>;
    using OutLayout = std::tuple_element_t<1, Tuple>;
    using WeiLayout = std::tuple_element_t<2, Tuple>;
    using InLayout  = std::tuple_element_t<3, Tuple>;

    std::vector<ck::utils::conv::ConvParam> conv_params;
    std::vector<ck::index_t> split_ks{1, 2};

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;
        for(auto split_k : split_ks)
        {
            for(auto& param : conv_params)
            {
                pass = pass && ck::profiler::profile_grouped_conv_bwd_data_impl<NDimSpatial,
                                                                                OutLayout,
                                                                                WeiLayout,
                                                                                InLayout,
                                                                                DataType,
                                                                                DataType,
                                                                                DataType>(
                                   true,  // do_verification
                                   1,     // init_method: integer value
                                   false, // do_log
                                   false, // time_kernel
                                   param,
                                   split_k);
            }
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes2d = ::testing::Types<std::tuple<float, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<ck::half_t, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<ck::bhalf_t, GNHWK, GKYXC, GNHWC>,
                                       std::tuple<float, NGKHW, GKYXC, NGCHW>,
                                       std::tuple<ck::half_t, NGKHW, GKYXC, NGCHW>,
                                       std::tuple<ck::bhalf_t, NGKHW, GKYXC, NGCHW>,
                                       std::tuple<float, NGKHW, GKCYX, NGCHW>,
                                       std::tuple<ck::half_t, NGKHW, GKCYX, NGCHW>,
                                       std::tuple<ck::bhalf_t, NGKHW, GKCYX, NGCHW>,
                                       std::tuple<float, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::half_t, NHWGK, GKYXC, NHWGC>,
                                       std::tuple<ck::bhalf_t, NHWGK, GKYXC, NHWGC>>;

using KernelTypes3d = ::testing::Types<std::tuple<float, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::half_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<ck::bhalf_t, GNDHWK, GKZYXC, GNDHWC>,
                                       std::tuple<float, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<ck::half_t, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<ck::bhalf_t, NGKDHW, GKZYXC, NGCDHW>,
                                       std::tuple<float, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<ck::half_t, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<ck::bhalf_t, NGKDHW, GKCZYX, NGCDHW>,
                                       std::tuple<float, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::half_t, NDHWGK, GKZYXC, NDHWGC>,
                                       std::tuple<ck::bhalf_t, NDHWGK, GKZYXC, NDHWGC>>;

template <typename Tuple>
class TestGroupedConvndBwdData2d : public TestGroupedConvndBwdData<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndBwdData3d : public TestGroupedConvndBwdData<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdData2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndBwdData3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdData2d, Test2D)
{
    this->conv_params.clear();
    // SplitN case
    this->conv_params.push_back(
        {2, 1, 128, 4, 192, {2, 2}, {224, 224}, {224, 224}, {1, 1}, {0, 0}, {0, 0}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndBwdData3d, Test3D)
{
    this->conv_params.clear();
    // SplitN case
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
    this->template Run<3>();
}
