// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test: Verify InstanceTraits works for Reference kernels

#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/reflect/instance_traits_reference.hpp"
#include "impl/conv_algorithm_types.hpp"
#include "impl/conv_signature_types.hpp"
#include <gtest/gtest.h>

namespace {

using namespace ck_tile::builder;
using namespace ck_tile::builder::test;

TEST(ReferenceInstanceTraits, Forward_2D_FP16)
{
    // Create a reference forward kernel
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::FORWARD,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    // Use InstanceTraits to query properties
    using Traits = ck_tile::reflect::InstanceTraits<RefKernel>;

    // Verify spatial dimension
    EXPECT_EQ(Traits::kSpatialDim, 2);

    // Verify direction
    EXPECT_EQ(Traits::direction, ConvDirection::FORWARD);

    // Verify data types
    EXPECT_TRUE((std::is_same_v<Traits::ADataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<Traits::BDataType, ck::half_t>));
    EXPECT_TRUE((std::is_same_v<Traits::EDataType, ck::half_t>));

    // Verify layouts
    EXPECT_TRUE((std::is_same_v<Traits::InLayout, ck::tensor_layout::convolution::NHWGC>));
    EXPECT_TRUE((std::is_same_v<Traits::WeiLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<Traits::OutLayout, ck::tensor_layout::convolution::NHWGK>));

    // Verify elementwise operations (always PassThrough for reference)
    EXPECT_TRUE(
        (std::is_same_v<Traits::AElementwiseOperation, ck_tile::element_wise::PassThrough>));
    EXPECT_TRUE(
        (std::is_same_v<Traits::BElementwiseOperation, ck_tile::element_wise::PassThrough>));
    EXPECT_TRUE(
        (std::is_same_v<Traits::CDEElementwiseOperation, ck_tile::element_wise::PassThrough>));

    // Verify block size is 0 (N/A for reference)
    EXPECT_EQ(Traits::kBlockSize, 0);

    // Verify instance_string() - now includes data type and layouts!
    std::string instance_str = Traits::instance_string();
    EXPECT_EQ(instance_str, "GPU_Reference_Forward_2D_fp16_NHWGC_GKYXC_NHWGK");

    std::cout << "✓ Forward InstanceTraits validated: " << instance_str << std::endl;
}

TEST(ReferenceInstanceTraits, BackwardData_2D_FP16)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_DATA,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    using Traits = ck_tile::reflect::InstanceTraits<RefKernel>;

    EXPECT_EQ(Traits::kSpatialDim, 2);
    EXPECT_EQ(Traits::direction, ConvDirection::BACKWARD_DATA);

    std::string instance_str = Traits::instance_string();
    EXPECT_EQ(instance_str, "GPU_Reference_BackwardData_2D_fp16_NHWGC_GKYXC_NHWGK");

    std::cout << "✓ Backward Data InstanceTraits validated: " << instance_str << std::endl;
}

TEST(ReferenceInstanceTraits, BackwardWeight_2D_FP16)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_WEIGHT,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::NHWGC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::NHWGK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    using Traits = ck_tile::reflect::InstanceTraits<RefKernel>;

    EXPECT_EQ(Traits::kSpatialDim, 2);
    EXPECT_EQ(Traits::direction, ConvDirection::BACKWARD_WEIGHT);

    std::string instance_str = Traits::instance_string();
    EXPECT_EQ(instance_str, "GPU_Reference_BackwardWeight_2D_fp16_NHWGC_GKYXC_NHWGK");

    std::cout << "✓ Backward Weight InstanceTraits validated: " << instance_str << std::endl;
}

} // namespace
