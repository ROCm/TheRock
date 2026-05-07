// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "profiler/profile_grouped_conv_fwd_impl.hpp"

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;
using I8                          = int8_t;
using F8                          = ck::f8_t;
using BF8                         = ck::bf8_t;
using F16                         = ck::half_t;
using BF16                        = ck::bhalf_t;
using F32                         = float;

template <typename Tuple>
class TestGroupedConvndFwd : public ::testing::Test
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
        for(size_t i = 0; i < conv_params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            // FP8 workaround for CDNA1/2 is currently broken, do not test.
            if(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a")
            {
                if(std::is_same<InDataType, F8>::value || std::is_same<InDataType, BF8>::value)
                {
                    printf("Skipping FP8 / BF8 tests on CDNA1/2.\n");
                    continue;
                }
            }
            auto& param = conv_params[i];
            pass        = pass && ck::profiler::profile_grouped_conv_fwd_impl<NDimSpatial,
                                                                              InLayout,
                                                                              WeiLayout,
                                                                              OutLayout,
                                                                              InDataType,
                                                                              WeiDataType,
                                                                              OutDataType,
                                                                              AComputeType,
                                                                              BComputeType,
                                                                              IndexType>(
                               2,     // do_verification
                               1,     // init_method: integer value
                               false, // do_log
                               false, // time_kernel
                               param,
                               ck::tensor_operation::element_wise::PassThrough{},
                               instance_index);
        }
        EXPECT_TRUE(pass);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes1d = ::testing::Types<std::tuple<F32, F32, F32, F32, F32, GNWC, GKXC, GNWK>,
                                       std::tuple<F16, F16, F16, F16, F16, GNWC, GKXC, GNWK>,
                                       std::tuple<BF16, BF16, BF16, BF16, BF16, GNWC, GKXC, GNWK>,
                                       std::tuple<I8, I8, I8, I8, I8, GNWC, GKXC, GNWK>>;

using KernelTypes2d =
    ::testing::Types<std::tuple<F32, F32, F32, F32, F32, GNHWC, GKYXC, GNHWK>,
                     std::tuple<F16, F16, F16, F16, F16, GNHWC, GKYXC, GNHWK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, GNHWC, GKYXC, GNHWK>,
                     std::tuple<F32, F32, F32, F32, F32, NHWGC, GKYXC, NHWGK>,
                     std::tuple<F16, F16, F16, F16, F16, NHWGC, GKYXC, NHWGK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NHWGC, GKYXC, NHWGK>,
                     std::tuple<I8, I8, I8, I8, I8, NHWGC, GKYXC, NHWGK>,
                     std::tuple<F32, F32, F32, F32, F32, NGCHW, GKYXC, NGKHW>,
                     std::tuple<F16, F16, F16, F16, F16, NGCHW, GKYXC, NGKHW>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NGCHW, GKYXC, NGKHW>,
                     std::tuple<I8, I8, I8, I8, I8, NGCHW, GKYXC, NGKHW>,
                     std::tuple<F32, F32, F32, F32, F32, NGCHW, GKCYX, NGKHW>,
                     std::tuple<F16, F16, F16, F16, F16, NGCHW, GKCYX, NGKHW>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NGCHW, GKCYX, NGKHW>>;

using KernelTypes3d =
    ::testing::Types<std::tuple<F32, F32, F32, F32, F32, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<F16, F16, F16, F16, F16, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<I8, I8, I8, I8, I8, GNDHWC, GKZYXC, GNDHWK>,
                     std::tuple<I8, I8, I8, I8, I8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F8, F8, F8, F8, F8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF8, BF8, F8, BF8, BF8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F8, BF8, F8, F8, BF8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF8, F8, F8, BF8, F8, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F32, F32, F32, F32, F32, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F16, F16, F16, F16, F16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NDHWGC, GKZYXC, NDHWGK>,
                     std::tuple<F32, F32, F32, F32, F32, NGCDHW, GKCZYX, NGKDHW>,
                     std::tuple<F16, F16, F16, F16, F16, NGCDHW, GKCZYX, NGKDHW>,
                     std::tuple<BF16, BF16, BF16, BF16, BF16, NGCDHW, GKCZYX, NGKDHW>>;

template <typename Tuple>
class TestGroupedConvndFwd1d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd2d : public TestGroupedConvndFwd<Tuple>
{
};

template <typename Tuple>
class TestGroupedConvndFwd3d : public TestGroupedConvndFwd<Tuple>
{
};

TYPED_TEST_SUITE(TestGroupedConvndFwd1d, KernelTypes1d);
TYPED_TEST_SUITE(TestGroupedConvndFwd2d, KernelTypes2d);
TYPED_TEST_SUITE(TestGroupedConvndFwd3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndFwd1d, Test1D)
{
    this->conv_params.clear();
    this->conv_params.push_back({1, 2, 32, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 2, 32, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 2, 32, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});
    this->conv_params.push_back({1, 1, 1, 1, 32, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 1, 1, 64, 3, {3}, {32}, {1}, {1}, {1}, {1}});
    this->conv_params.push_back({1, 96, 1, 1, 1, {3}, {512}, {1}, {1}, {1}, {1}});
    this->template Run<1>();
}

TYPED_TEST(TestGroupedConvndFwd2d, Test2D)
{
    this->conv_params.clear();
    this->conv_params.push_back(
        {2, 3, 5, 96, 200, {1, 1}, {73, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {1, 1}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {2, 2}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {3, 3}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {5, 5}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 1, 1, 32, 32, {9, 9}, {128, 128}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});

    this->conv_params.push_back(
        {2, 2, 32, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    this->conv_params.push_back({2, 1, 1, 1, 32, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 64, 3, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back({2, 1, 1, 1, 1, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});

    this->conv_params.push_back(
        {2, 96, 1, 1, 1, {1, 1}, {120, 160}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->conv_params.push_back(
        {2, 96, 1, 1, 1, {3, 3}, {120, 160}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    this->template Run<2>();
}

TYPED_TEST(TestGroupedConvndFwd3d, Test3D)
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
