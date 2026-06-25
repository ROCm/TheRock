// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle.hpp"

#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#include <gtest/gtest.h>

using F16         = ck::half_t;
using F32         = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using ConvolutionBackwardWeightSpecialization =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization;

static constexpr auto ConvBwdWeightDefault = ConvolutionBackwardWeightSpecialization::Default;
static constexpr auto Filter1x1Stride1Pad0 =
    ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

template <typename Tuple, ConvolutionBackwardWeightSpecialization ConvSpec>
class TestGroupedConvndBwdWeight : public ::testing::Test
{
    protected:
    using OutLayout                          = std::tuple_element_t<0, Tuple>;
    using WeiLayout                          = std::tuple_element_t<1, Tuple>;
    using InLayout                           = std::tuple_element_t<2, Tuple>;
    static constexpr ck::index_t NDimSpatial = std::tuple_element_t<3, Tuple>{};

    // clang-format off
    using GroupedConvBwdWeightDeviceInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffle
        //|    NumDim|       A|       B|       C| AData| BData|  CData| AccData|            A|           B|            C|    ConvForward| Block|  MPer|  NPer|  KPer| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|       CShuffle|       CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //|   Spatial|  Layout|  Layout|  Layout|  Type|  Type|   Type|    Type|  Elementwise| Elementwise|  Elementwise| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MRepeatPerWave| NRepeatPerWave|            _MBlock_MPerBlock| ScalarPerVector|
        //|          |        |        |        |      |      |       |        |    Operation|   Operation|    Operation|               |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |     PerShuffle|     PerShuffle|            _NBlock_NPerBlock|      _NPerBlock|
        //|          |        |        |        |      |      |       |        |             |            |             |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |               |               |                             |                |
    <NDimSpatial, InLayout, WeiLayout, OutLayout,  F16,   F16,  F16,  F32,  PassThrough, PassThrough, PassThrough,       ConvSpec,          128,   128,   128,     8,  8,    16,   16,       4,       4,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>;
    // clang-format on

    ck::utils::conv::ConvParam conv_param;

    void SetUp() override
    {
        if(!ck::is_gfx11_supported())
        {
            GTEST_SKIP();
        }
    }

    template <ck::index_t SplitK>
    bool Run()
    {

        const auto in_g_n_c_wis_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(
                conv_param);

        const auto wei_g_k_c_xs_desc =
            ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(
                conv_param);

        const auto out_g_n_k_wos_desc =
            ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(
                conv_param);

        std::array<ck::index_t, NDimSpatial + 3> input_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> filter_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> output_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> input_strides{};
        std::array<ck::index_t, NDimSpatial + 3> weights_strides{};
        std::array<ck::index_t, NDimSpatial + 3> output_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto range_copy = [](const auto& from, auto to) { std::copy(begin(from), end(from), to); };

        range_copy(in_g_n_c_wis_desc.GetLengths(), begin(input_lengths));
        range_copy(in_g_n_c_wis_desc.GetStrides(), begin(input_strides));
        range_copy(wei_g_k_c_xs_desc.GetLengths(), begin(filter_lengths));
        range_copy(wei_g_k_c_xs_desc.GetStrides(), begin(weights_strides));
        range_copy(out_g_n_k_wos_desc.GetLengths(), begin(output_lengths));
        range_copy(out_g_n_k_wos_desc.GetStrides(), begin(output_strides));
        range_copy(conv_param.conv_filter_strides_, begin(conv_filter_strides));
        range_copy(conv_param.conv_filter_dilations_, begin(conv_filter_dilations));
        range_copy(conv_param.input_left_pads_, begin(input_left_pads));
        range_copy(conv_param.input_right_pads_, begin(input_right_pads));

        auto conv = GroupedConvBwdWeightDeviceInstance{};

        auto argument = conv.MakeArgument(nullptr,
                                          nullptr,
                                          nullptr,
                                          input_lengths,
                                          input_strides,
                                          filter_lengths,
                                          weights_strides,
                                          output_lengths,
                                          output_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          PassThrough{},
                                          PassThrough{},
                                          PassThrough{},
                                          SplitK);
        return conv.IsSupportedArgument(argument);
    }
};

using namespace ck::tensor_layout::convolution;

using KernelTypes3d = ::testing::Types<std::tuple<GNDHWK, GKZYXC, GNDHWC, ck::Number<3>>,
                                       std::tuple<NDHWGK, GKZYXC, NDHWGC, ck::Number<3>>>;

template <typename Tuple>
class TestGroupedConvndBwdWeightFilter1x13d
    : public TestGroupedConvndBwdWeight<Tuple, Filter1x1Stride1Pad0>
{
};

template <typename Tuple>
class TestGroupedConvndBwdWeightDefault3d
    : public TestGroupedConvndBwdWeight<Tuple, ConvBwdWeightDefault>
{
};

TYPED_TEST_SUITE(TestGroupedConvndBwdWeightFilter1x13d, KernelTypes3d);
TYPED_TEST_SUITE(TestGroupedConvndBwdWeightDefault3d, KernelTypes3d);

TYPED_TEST(TestGroupedConvndBwdWeightFilter1x13d, SpecializationCheck)
{
    // Check filter 3x3x3 instead of 1x1x1
    this->conv_param = {
        3, 2, 4, 192, 192, {3, 3, 3}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    bool is_supported = this->template Run<1>();
    EXPECT_FALSE(is_supported);

    // Check strides 2x2x2 instead of 1x1x1
    this->conv_param = {
        3, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    is_supported = this->template Run<1>();
    EXPECT_FALSE(is_supported);

    // Check with pad
    this->conv_param = {
        3, 2, 4, 192, 192, {1, 1, 1}, {28, 28, 28}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    is_supported = this->template Run<1>();
    EXPECT_FALSE(is_supported);

    // Supported version
    this->conv_param = {
        3, 2, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    is_supported = this->template Run<1>();
    EXPECT_TRUE(is_supported);
}

TYPED_TEST(TestGroupedConvndBwdWeightDefault3d, VectorLoadCheck)
{
    // vector load for A
    this->conv_param = {
        3, 2, 128, 129, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    bool is_supported = this->template Run<1>();
    EXPECT_FALSE(is_supported);
    // vector load for B, E, Ds
    this->conv_param = {
        3, 2, 128, 128, 257, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    is_supported = this->template Run<1>();
    EXPECT_FALSE(is_supported);
}

TYPED_TEST(TestGroupedConvndBwdWeightDefault3d, SplitKCheck)
{
    // SplitK=1
    this->conv_param = {
        3, 2, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    bool is_supported = this->template Run<1>();
    EXPECT_TRUE(is_supported);
    // SplitK=2
    this->conv_param = {
        3, 2, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}};
    is_supported = this->template Run<2>();
    EXPECT_FALSE(is_supported);
}
