// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#include <gtest/gtest.h>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using ScaleAdd    = ck::tensor_operation::element_wise::ScaleAdd;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename DataType,
          typename InDataTypes,
          typename WeiDataTypes,
          typename InElementOp,
          typename WeiElementOp>
class TestGroupedConvndFwdMultiABInterfaceBase : public ::testing::Test
{
    protected:
    static constexpr ck::index_t NDimSpatial = 3;
    static constexpr ck::index_t NumAs       = 2;
    static constexpr ck::index_t NumBs       = 2;
    static constexpr auto ConvSpec =
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    using InLayout                 = ck::tensor_layout::convolution::GNDHWC;
    using WeiLayout                = ck::tensor_layout::convolution::GKZYXC;
    using OutLayout                = ck::tensor_layout::convolution::GNDHWK;
    using OutElementOp             = PassThrough;

    using DeviceGroupedConvNDMultiABFwdInstance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
            NDimSpatial,
            InLayout,
            WeiLayout,
            ck::Tuple<>,
            OutLayout,
            InDataTypes,
            WeiDataTypes,
            DataType,
            DataType,
            ck::Tuple<>,
            DataType,
            InElementOp,
            WeiElementOp,
            OutElementOp,
            ConvSpec,    // ConvForwardSpecialization
            GemmSpec,    // GemmSpecialization
            1,           //
            256,         // BlockSize
            128,         // MPerBlock
            256,         // NPerBlock
            32,          // KPerBlock
            8,           // AK1
            8,           // BK1
            16,          // MPerXdl
            16,          // NPerXdl
            4,           // MXdlPerWave
            8,           // NXdlPerWave
            S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
            S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
            S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
            2,           // ABlockTransferSrcVectorDim
            8,           // ABlockTransferSrcScalarPerVector
            8,           // ABlockTransferDstScalarPerVector_AK1
            1,           // ABlockLdsExtraM
            S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
            S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
            S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
            2,           // BBlockTransferSrcVectorDim
            8,           // BBlockTransferSrcScalarPerVector
            8,           // BBlockTransferDstScalarPerVector_BK1
            1,           // BBlockLdsExtraN
            1,
            1,
            S<1, 32, 1, 8>,
            4>;

    const ck::utils::conv::ConvParam conv_param{
        3, 1, 16, 16, 8, {3, 3, 3}, {17, 17, 17}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    void SetUp() override
    {
        if(!ck::is_xdl_supported())
        {
            GTEST_SKIP();
        }
    }

    template <typename ADataType, typename BDataType>
    bool Run(ADataType as, BDataType bs)
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

        std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
        std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
        std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

        copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
        copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
        copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
        copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
        copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
        copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        std::array<const void*, 0> ds{};

        // do Conv
        auto conv     = DeviceGroupedConvNDMultiABFwdInstance{};
        auto invoker  = conv.MakeInvoker();
        auto argument = conv.MakeArgument(as,
                                          bs,
                                          ds,
                                          nullptr,
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          {},
                                          {},
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          InElementOp{},
                                          WeiElementOp{},
                                          OutElementOp{});

        return conv.IsSupportedArgument(argument);
    }
};

class TestGroupedConvndFwdMultiAInterface
    : public TestGroupedConvndFwdMultiABInterfaceBase<float,
                                                      ck::Tuple<ck::half_t, ck::half_t>,
                                                      ck::half_t,
                                                      ScaleAdd,
                                                      PassThrough>
{
};

class TestGroupedConvndFwdMultiBInterface
    : public TestGroupedConvndFwdMultiABInterfaceBase<float,
                                                      ck::half_t,
                                                      ck::Tuple<ck::half_t, ck::half_t>,
                                                      PassThrough,
                                                      ScaleAdd>
{
};

class TestGroupedConvndFwdMultiABInterface
    : public TestGroupedConvndFwdMultiABInterfaceBase<float,
                                                      ck::Tuple<ck::half_t, ck::half_t>,
                                                      ck::Tuple<ck::half_t, ck::half_t>,
                                                      ScaleAdd,
                                                      ScaleAdd>
{
};

class TestGroupedConvndFwdInterface : public TestGroupedConvndFwdMultiABInterfaceBase<float,
                                                                                      ck::half_t,
                                                                                      ck::half_t,
                                                                                      PassThrough,
                                                                                      PassThrough>
{
};

TEST_F(TestGroupedConvndFwdMultiAInterface, MultiA)
{
    std::array<const void*, NumAs> as{nullptr, nullptr};
    const void* b = nullptr;

    EXPECT_TRUE(this->Run(as, b));
}

TEST_F(TestGroupedConvndFwdMultiBInterface, MultiB)
{
    const void* a = nullptr;
    std::array<const void*, NumBs> bs{nullptr, nullptr};

    EXPECT_TRUE(this->Run(a, bs));
}

TEST_F(TestGroupedConvndFwdMultiABInterface, MultiAB)
{
    std::array<const void*, NumAs> as{nullptr, nullptr};
    std::array<const void*, NumBs> bs{nullptr, nullptr};

    EXPECT_TRUE(this->Run(as, bs));
}

TEST_F(TestGroupedConvndFwdInterface, SingleAB)
{
    const void* a = nullptr;
    const void* b = nullptr;

    EXPECT_TRUE(this->Run(a, b));
}
