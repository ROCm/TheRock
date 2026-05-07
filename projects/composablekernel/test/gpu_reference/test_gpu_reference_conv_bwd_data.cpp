// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "gpu_reference_utils.hpp"

using namespace ck;
using ck::test::ConvKernelType;

TEST(GpuReferenceConvBwdData, Conv2DFP16Small)
{
    auto params = test::conv_test_shapes::get_2d_small();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP32Medium)
{
    auto params = test::conv_test_shapes::get_2d_medium();
    bool result =
        test::test_conv_gpu_ref<2, float, float, float>(params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv1DFP16)
{
    auto params = test::conv_test_shapes::get_1d();
    bool result =
        test::test_conv_gpu_ref<1, half_t, half_t, half_t>(params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv3DFP16Small)
{
    auto params = test::conv_test_shapes::get_3d_small();
    bool result =
        test::test_conv_gpu_ref<3, half_t, half_t, half_t>(params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP16Stride2)
{
    auto params = test::conv_test_shapes::get_2d_stride2();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP16GroupedG2)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result =
        test::test_conv_gpu_ref<2, half_t, half_t, half_t>(params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP32GroupedG4)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g4();
    bool result =
        test::test_conv_gpu_ref<2, float, float, float>(params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP32GroupedNHWGC_GKYXC_NHWGK)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test::test_conv_gpu_ref<2,
                                          float,
                                          float,
                                          float,
                                          tensor_layout::convolution::NHWGC,
                                          tensor_layout::convolution::GKYXC,
                                          tensor_layout::convolution::NHWGK>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP16GroupedNHWGC_GKYXC_NHWGK)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test::test_conv_gpu_ref<2,
                                          half_t,
                                          half_t,
                                          half_t,
                                          tensor_layout::convolution::NHWGC,
                                          tensor_layout::convolution::GKYXC,
                                          tensor_layout::convolution::NHWGK>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP32GroupedNGCHW_GKYXC_NGKHW)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test::test_conv_gpu_ref<2,
                                          float,
                                          float,
                                          float,
                                          tensor_layout::convolution::NGCHW,
                                          tensor_layout::convolution::GKYXC,
                                          tensor_layout::convolution::NGKHW>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP16GroupedNGCHW_GKYXC_NGKHW)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test::test_conv_gpu_ref<2,
                                          half_t,
                                          half_t,
                                          half_t,
                                          tensor_layout::convolution::NGCHW,
                                          tensor_layout::convolution::GKYXC,
                                          tensor_layout::convolution::NGKHW>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP32GroupedNGCHW_GKCYX_NGKHW)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test::test_conv_gpu_ref<2,
                                          float,
                                          float,
                                          float,
                                          tensor_layout::convolution::NGCHW,
                                          tensor_layout::convolution::GKCYX,
                                          tensor_layout::convolution::NGKHW>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv2DFP16GroupedNGCHW_GKCYX_NGKHW)
{
    auto params = test::conv_test_shapes::get_2d_grouped_g2();
    bool result = test::test_conv_gpu_ref<2,
                                          half_t,
                                          half_t,
                                          half_t,
                                          tensor_layout::convolution::NGCHW,
                                          tensor_layout::convolution::GKCYX,
                                          tensor_layout::convolution::NGKHW>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv3DFP32GroupedNDHWGC_GKZYXC_NDHWGK)
{
    auto params = test::conv_test_shapes::get_3d_small();
    // Modify to be grouped (G=2)
    params.G_ = 2;
    params.C_ = 16; // 8 per group
    params.K_ = 16; // 8 per group

    bool result = test::test_conv_gpu_ref<3,
                                          float,
                                          float,
                                          float,
                                          tensor_layout::convolution::NDHWGC,
                                          tensor_layout::convolution::GKZYXC,
                                          tensor_layout::convolution::NDHWGK>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv3DFP16GroupedNDHWGC_GKZYXC_NDHWGK)
{
    auto params = test::conv_test_shapes::get_3d_small();
    // Modify to be grouped (G=2)
    params.G_ = 2;
    params.C_ = 16; // 8 per group
    params.K_ = 16; // 8 per group

    bool result = test::test_conv_gpu_ref<3,
                                          half_t,
                                          half_t,
                                          half_t,
                                          tensor_layout::convolution::NDHWGC,
                                          tensor_layout::convolution::GKZYXC,
                                          tensor_layout::convolution::NDHWGK>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv3DFP32GroupedNGCDHW_GKCZYX_NGKDHW)
{
    auto params = test::conv_test_shapes::get_3d_small();
    // Modify to be grouped (G=2)
    params.G_ = 2;
    params.C_ = 16; // 8 per group
    params.K_ = 16; // 8 per group

    bool result = test::test_conv_gpu_ref<3,
                                          float,
                                          float,
                                          float,
                                          tensor_layout::convolution::NGCDHW,
                                          tensor_layout::convolution::GKCZYX,
                                          tensor_layout::convolution::NGKDHW>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}

TEST(GpuReferenceConvBwdData, Conv3DFP16GroupedNGCDHW_GKCZYX_NGKDHW)
{
    auto params = test::conv_test_shapes::get_3d_small();
    // Modify to be grouped (G=2)
    params.G_ = 2;
    params.C_ = 16; // 8 per group
    params.K_ = 16; // 8 per group

    bool result = test::test_conv_gpu_ref<3,
                                          half_t,
                                          half_t,
                                          half_t,
                                          tensor_layout::convolution::NGCDHW,
                                          tensor_layout::convolution::GKCZYX,
                                          tensor_layout::convolution::NGKDHW>(
        params, ConvKernelType::BackwardData);
    EXPECT_TRUE(result);
}
