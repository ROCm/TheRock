// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_grouped_conv_fwd_bilinear_impl.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

using I8   = int8_t;
using F8   = ck::f8_t;
using BF8  = ck::bf8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

template <typename Tuple>
class TestGroupedConvndFwdBilinear : public ::testing::Test
{
    protected:
    using InDataType   = std::tuple_element_t<0, Tuple>;
    using WeiDataType  = std::tuple_element_t<1, Tuple>;
    using OutDataType  = std::tuple_element_t<2, Tuple>;
    using AComputeType = std::tuple_element_t<3, Tuple>;
    using BComputeType = std::tuple_element_t<4, Tuple>;
    using InLayout     = std::tuple_element_t<5, Tuple>;
    using WeiLayout    = std::tuple_element_t<6, Tuple>;
    using OutLayout    = std::tuple_element_t<7, Tuple>;
    using IndexType    = ck::index_t;

    std::vector<ck::utils::conv::ConvParam> conv_params;

    template <ck::index_t NDimSpatial>
    void Run()
    {
        EXPECT_FALSE(conv_params.empty());
        bool pass = true;

        // Create a Bilinear operation (binary element-wise operation)
        const auto bilinear_op = ck::tensor_operation::element_wise::Bilinear{};

        for(auto& param : conv_params)
        {
            if(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a")
            {
                if(std::is_same<InDataType, F8>::value || std::is_same<InDataType, BF8>::value)
                {
                    printf("Skipping FP8 / BF8 tests on CDNA1/2.\n");
                    continue;
                }
            }
            pass = pass && ck::profiler::profile_grouped_conv_fwd_bilinear_impl<
                               NDimSpatial,
                               InLayout,
                               WeiLayout,
                               OutLayout, // D layout same as output
                               OutLayout,
                               InDataType,
                               WeiDataType,
                               OutDataType, // D data type same as output
                               OutDataType,
                               AComputeType,
                               BComputeType,
                               IndexType>(true,  // do_verification
                                          1,     // init_method: integer value
                                          false, // do_log
                                          true,  // time_kernel
                                          param,
                                          bilinear_op);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes3d =
    ::testing::Types<std::tuple<F16, F16, F16, F16, F16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F32, F32, F32, F32, F32, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<I8, I8, I8, I8, I8, NDHWGC, GKZYXC, NDHWGK>>;

template <typename Tuple>
class TestGroupedConvndFwdBilinear3d : public TestGroupedConvndFwdBilinear<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwdBilinear3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwdBilinear3d, Test3D)
{
    this->conv_params.clear();

    this->conv_params.push_back(
        {3, 3, 5, 96, 200, {1, 1, 1}, {37, 37, 16}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {1, 1, 1}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {2, 2, 2}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {5, 5, 5}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {9, 9, 9}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {3, 3, 3}, {14, 14, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    this->conv_params.push_back(
        {3, 2, 32, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    this->conv_params.push_back(
        {3, 1, 1, 32, 32, {1, 1, 1}, {16, 16, 16}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    this->conv_params.push_back(
        {3, 1, 1, 1, 32, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 64, 3, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 1, 1, 1, 1, {3, 3, 3}, {32, 32, 32}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});

    this->conv_params.push_back(
        {3, 96, 1, 1, 1, {1, 1, 1}, {120, 40, 20}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->conv_params.push_back(
        {3, 96, 1, 1, 1, {3, 3, 3}, {120, 40, 20}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    this->template Run<3>();
}
