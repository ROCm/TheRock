// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_image_to_column_impl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_column_to_image_impl.hpp"
#include "ck/tensor_operation/gpu/device/conv_tensor_rearrange_op.hpp"

#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#include <gtest/gtest.h>

using ::ck::HostTensorDescriptor;

using DataType = float;
using ImLayout = ck::tensor_layout::convolution::GNWC;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::conv_tensor_rearrange_op;

template <ck::index_t ScalarPerVector, bool IsCPacked>
class TestConvTensorRearrangeInterface : public ::testing::Test
{
    protected:
    static constexpr ck::index_t NDimSpatial = 1;

    // clang-format off
    using DeviceImgToColInstance = ck::tensor_operation::device::DeviceImageToColumnImpl
        //        Num| ImLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread| Scalar|
        //        Dim|         |           |            |  Size| Block| Block|   Cluster|    Per|
        //    Spatial|         |           |            |      |      |      |   Lengths| Vector|
        //           |         |           |            |      |      |      |          |       |
        < NDimSpatial, ImLayout,   DataType,    DataType,   256,   128,   128, S<16, 16>,ScalarPerVector>;
    using DeviceColToimgInstance = ck::tensor_operation::device::DeviceColumnToImageImpl
        //        Num| ImLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread| Scalar|
        //        Dim|         |           |            |  Size| Block| Block|   Cluster|    Per|
        //    Spatial|         |           |            |      |      |      |   Lengths| Vector|
        //           |         |           |            |      |      |      |          |       |
        < NDimSpatial, ImLayout,   DataType,    DataType,   256,   128,   128, S<16, 16>,ScalarPerVector>;
    // clang-format on

    ck::utils::conv::ConvParam conv_param;

    template <typename ConvTensorRearrangeOp>
    bool Run()
    {
        const auto G = conv_param.G_;
        const auto N = conv_param.N_;
        const auto C = conv_param.C_;
        const auto FakeC =
            conv_param.C_ / 2; // Fake C to simulate the behavior that C is not packed

        const ck::index_t NDoHoWo =
            N *
            ck::accumulate_n<ck::index_t>(
                conv_param.output_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());
        const ck::index_t CZYX =
            C *
            ck::accumulate_n<ck::index_t>(
                conv_param.filter_spatial_lengths_.begin(), NDimSpatial, 1, std::multiplies<>());

        const auto image_desc =
            ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<ImLayout>(
                conv_param);
        const auto gemm_desc = HostTensorDescriptor({G, NDoHoWo, CZYX});

        std::array<ck::index_t, NDimSpatial> input_spatial_lengths{};
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths{};
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths{};
        std::array<ck::index_t, NDimSpatial + 3> input_g_n_c_wis_strides{};
        std::array<ck::index_t, 3> output_g_m_k_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
        std::array<ck::index_t, NDimSpatial> input_left_pads{};
        std::array<ck::index_t, NDimSpatial> input_right_pads{};

        auto copy = [](const auto& x, auto& y) { std::copy(x.begin(), x.end(), y.begin()); };

        copy(conv_param.input_spatial_lengths_, input_spatial_lengths);
        copy(conv_param.filter_spatial_lengths_, filter_spatial_lengths);
        copy(conv_param.output_spatial_lengths_, output_spatial_lengths);
        copy(image_desc.GetStrides(), input_g_n_c_wis_strides);
        copy(gemm_desc.GetStrides(), output_g_m_k_strides);
        copy(conv_param.conv_filter_strides_, conv_filter_strides);
        copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
        copy(conv_param.input_left_pads_, input_left_pads);
        copy(conv_param.input_right_pads_, input_right_pads);

        if constexpr(std::is_same_v<ConvTensorRearrangeOp, ImageToColumn>)
        {
            auto img2col  = DeviceImgToColInstance{};
            auto argument = img2col.MakeArgument(nullptr,
                                                 nullptr,
                                                 G,
                                                 N,
                                                 IsCPacked ? C : FakeC,
                                                 input_spatial_lengths,
                                                 filter_spatial_lengths,
                                                 output_spatial_lengths,
                                                 input_g_n_c_wis_strides,
                                                 output_g_m_k_strides,
                                                 conv_filter_strides,
                                                 conv_filter_dilations,
                                                 input_left_pads,
                                                 input_right_pads);

            return img2col.IsSupportedArgument(argument);
        }
        else if constexpr(std::is_same_v<ConvTensorRearrangeOp, ColumnToImage>)
        {
            auto col2img  = DeviceColToimgInstance{};
            auto argument = col2img.MakeArgument(nullptr,
                                                 nullptr,
                                                 G,
                                                 N,
                                                 IsCPacked ? C : FakeC,
                                                 input_spatial_lengths,
                                                 filter_spatial_lengths,
                                                 output_spatial_lengths,
                                                 input_g_n_c_wis_strides,
                                                 output_g_m_k_strides,
                                                 conv_filter_strides,
                                                 conv_filter_dilations,
                                                 input_left_pads,
                                                 input_right_pads);

            return col2img.IsSupportedArgument(argument);
        }
        throw std::runtime_error("Conv_tensor_rearrange: problem with tensor rearrange operator. ");
        return 1;
    }
};

class TestConvTensorRearrangeInterface1ScalarPerVector
    : public TestConvTensorRearrangeInterface<1, true>
{
};

class TestConvTensorRearrangeInterface4ScalarPerVector
    : public TestConvTensorRearrangeInterface<4, true>
{
};

class TestConvTensorRearrangeInterface4ScalarPerVectorFakeC
    : public TestConvTensorRearrangeInterface<4, false>
{
};

TEST_F(TestConvTensorRearrangeInterface1ScalarPerVector, X1ScalarPerVector)
{
    // vector load C * X % ScalarPerVector
    this->conv_param  = {1, 1, 1, 1, 1, {3}, {3}, {1}, {1}, {0}, {0}};
    bool is_supported = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
    // vector load C * left_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {3}, {0}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
    // vector load C * right_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {0}, {3}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
    // vector load C % ScalarPerVector, right_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {0}, {3}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
    // vector load C % ScalarPerVector, left_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {3}, {0}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
    // vector load C % ScalarPerVector, dilation
    this->conv_param = {1, 1, 1, 1, 1, {4}, {8}, {1}, {2}, {0}, {0}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
    // C = 4
    this->conv_param = {1, 1, 1, 1, 4, {3}, {3}, {1}, {1}, {3}, {3}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
}

TEST_F(TestConvTensorRearrangeInterface4ScalarPerVector, X4ScalarPerVector)
{
    // vector load C * X % ScalarPerVector
    this->conv_param  = {1, 1, 1, 1, 1, {3}, {3}, {1}, {1}, {0}, {0}};
    bool is_supported = this->template Run<ImageToColumn>();
    EXPECT_FALSE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_FALSE(is_supported);
    // vector load C * left_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {3}, {0}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_FALSE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_FALSE(is_supported);
    // vector load C * right_pad_x % ScalarPerVector
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {1}, {1}, {0}, {3}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_FALSE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_FALSE(is_supported);
    // vector load C % ScalarPerVector, right_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {0}, {3}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_FALSE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_FALSE(is_supported);
    // vector load C % ScalarPerVector, left_pad and stride
    this->conv_param = {1, 1, 1, 1, 1, {4}, {3}, {2}, {1}, {3}, {0}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_FALSE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_FALSE(is_supported);
    // vector load C % ScalarPerVector, dilation
    this->conv_param = {1, 1, 1, 1, 1, {4}, {8}, {1}, {2}, {0}, {0}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_FALSE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_FALSE(is_supported);
    // C = 4
    this->conv_param = {1, 1, 1, 1, 4, {3}, {3}, {1}, {1}, {3}, {3}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
}

TEST_F(TestConvTensorRearrangeInterface4ScalarPerVectorFakeC, X4ScalarPerVectorFakeC)
{
    // C = 3
    this->conv_param  = {1, 1, 1, 1, 3, {4}, {5}, {1}, {1}, {0}, {0}};
    bool is_supported = this->template Run<ImageToColumn>();
    EXPECT_FALSE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_FALSE(is_supported);
    // C = 4
    this->conv_param = {1, 1, 1, 1, 8, {4}, {5}, {1}, {1}, {0}, {0}};
    is_supported     = this->template Run<ImageToColumn>();
    EXPECT_TRUE(is_supported);
    is_supported = this->template Run<ColumnToImage>();
    EXPECT_TRUE(is_supported);
}
