// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

namespace {

using namespace ck_tile::builder;
using namespace ck_tile::builder::test_utils;

TEST(FwdConvInstances,
     Create_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_Instance_2D_BF16_scale_add_relu)
{
    using enum ck_tile::builder::ConvDirection;
    using enum ck_tile::builder::DataType;
    using enum ck_tile::builder::TensorLayout;
    using enum ck_tile::builder::ElementwiseOperation;

    constexpr ConvSignature FwdConvSignature{
        .spatial_dim            = 2,
        .direction              = FORWARD,
        .data_type              = BF16,
        .accumulation_data_type = FP32,
        .input                  = {.config = {.layout = NHWGC}},
        .weight                 = {.config = {.layout = GKYXC, .data_type = BF16}},
        .output                 = ConvolutionTensor{
                            .config    = {.layout = NHWGK},
                            .operation = TensorOperation<>{.elementwise_operation = SCALEADD_SCALEADD_RELU}
                             .with_auxiliary_operand_configs<NHWGK, G_K_strided>()}};

    constexpr auto FwdConvAlgorithm =
        ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle{}
            .with_thread_block(FwdThreadBlock_64_64x32x32)
            .with_gemm_config(FwdGemmParams_Xdl_2x1_per_wave)
            .with_transfer(FwdTransfer_4x16x1)
            .with_specializations(ConvFwdSpecialization::DEFAULT, GemmSpecialization::MNKPadding)
            .with_prefetch_config(1, 1, PipelineScheduler::DEFAULT);

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    const auto expected_transfer_parameters = to_string(FwdConvAlgorithm);
    run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle",
                       expected_transfer_parameters,
                       "NHWGC,GKYXC,Tuple(NHWGK,G_K),NHWGK",
                       "PassThrough,PassThrough,ScaleAddScaleAddRelu",
                       "MNKPadding",
                       "Default"});
}

} // namespace
