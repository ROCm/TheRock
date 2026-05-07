// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/builder/testing/conv_fwd_ck.hpp"
#include "ck_tile/builder/testing/conv_fwd_reference.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "testing_utils.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

using ck_tile::test::MatchesReference;

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3{}
                               .with_thread_block(cku::FwdThreadBlock_256_256x256x32)
                               .with_gemm_config(cku::FwdGemmParams_Xdl_4x4_per_wave)
                               .with_transfer(cku::FwdTransfer_4x64x1)
                               .with_specializations(ckb::ConvFwdSpecialization::DEFAULT,
                                                     ckb::GemmSpecialization::MNKPadding)
                               .with_block_gemm(cku::BlockGemmDesc_v3_intrawave);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

using Reference = ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;

TEST(Fwd2DFp16_CShufV3_GNHWC, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    cku::run_test<Builder>({"DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
                            expected_transfer_parameters,
                            "Default",
                            "Intrawave",
                            "v3",
                            "GNHWC,GKYXC,EmptyTuple,GNHWK",
                            "PassThrough,PassThrough,PassThrough",
                            "MNKPadding"});
}

TEST(Fwd2DFp16_CShufV3_GNHWC, EndToEnd)
{
    if(!ck_tile::get_device_name().starts_with("gfx9"))
    {
        GTEST_SKIP() << "unsupported architecture";
    }

    ckt::Args<SIGNATURE> args = {
        .lengths =
            {
                .batch_size      = 16,
                .groups          = 1,
                .input_channels  = 32,
                .output_channels = 48,
                .image =
                    {
                        .width  = 56,
                        .height = 64,
                    },
                .filter =
                    {
                        .width  = 3,
                        .height = 5,
                    },
            },
        .filter_strides     = {.width = 1, .height = 1},
        .filter_dilation    = {.width = 1, .height = 1},
        .input_left_pad     = {.width = 0, .height = 0},
        .input_right_pad    = {.width = 0, .height = 0},
        .a_elementwise_op   = {},
        .b_elementwise_op   = {},
        .cde_elementwise_op = {},
    };

    auto inputs    = ckt::alloc_inputs(args);
    auto outputs   = ckt::alloc_outputs(args);
    auto reference = ckt::alloc_outputs(args);

    ckt::init_inputs(args, inputs.get());

    auto conv = Instance{};
    ckt::run(conv, args, inputs.get(), outputs.get());

    auto ref_conv = Reference{};
    ckt::run(ref_conv, args, inputs.get(), reference.get());

    EXPECT_THAT(outputs.get(), MatchesReference(args, reference.get()));
}
