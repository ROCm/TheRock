// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"

namespace {

using ::ck_tile::builder::ElementwiseOperation;
using ::ck_tile::builder::factory::internal::ElementwiseOpToCK;

TEST(ConvElementwiseOp, AssignsOpsForPassThrough)
{
    using Op = ElementwiseOpToCK<ElementwiseOperation::PASS_THROUGH>::Op;
    EXPECT_TRUE((std::is_same_v<Op, ck::tensor_operation::element_wise::PassThrough>));
}

TEST(ConvElementwiseOp, AssignsOpsForScale)
{
    using Op = ElementwiseOpToCK<ElementwiseOperation::SCALE>::Op;
    EXPECT_TRUE((std::is_same_v<Op, ck::tensor_operation::element_wise::Scale>));
}

TEST(ConvElementwiseOp, AssignsOpsForClamp)
{
    using Op = ElementwiseOpToCK<ElementwiseOperation::CLAMP>::Op;
    EXPECT_TRUE((std::is_same_v<Op, ck::tensor_operation::element_wise::Clamp>));
}

TEST(ConvElementwiseOp, AssignsOpsForScaleAddScaleAddRelu)
{
    using Op = ElementwiseOpToCK<ElementwiseOperation::SCALEADD_SCALEADD_RELU>::Op;
    EXPECT_TRUE((std::is_same_v<Op, ck::tensor_operation::element_wise::ScaleAddScaleAddRelu>));
}

TEST(ConvElementwiseOp, AssignsOpsForBiasNormClamp)
{
    using Op = ElementwiseOpToCK<ElementwiseOperation::BIAS_BNORM_CLAMP>::Op;
    EXPECT_TRUE(
        (std::is_same_v<Op, ck::tensor_operation::element_wise::BiasNormalizeInInferClamp>));
}

} // namespace
