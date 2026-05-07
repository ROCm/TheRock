// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include "impl/conv_signature_types.hpp"

namespace {

namespace ckb = ck_tile::builder;
using ck_tile::builder::DataType;
using ck_tile::builder::ElementwiseOperation;
using ck_tile::builder::TensorLayout;
using ck_tile::builder::factory::internal::AuxiliaryTensorLayouts;
using ck_tile::builder::factory::internal::ConvTensorLayouts;
using ck_tile::builder::factory::internal::LayoutToCK;
using ck_tile::builder::test::ConvolutionTensor;
using ck_tile::builder::test::ConvSignature;
using ck_tile::builder::test::TensorConfig;
using ck_tile::builder::test::TensorOperation;

namespace enums {
using enum ck_tile::builder::ConvDirection;
using enum ck_tile::builder::TensorLayout;
using enum ck_tile::builder::DataType;
} // namespace enums

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NWGC_GKXC_NWGK)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 1,
                                                .direction              = FORWARD,
                                                .data_type              = FP16,
                                                .accumulation_data_type = FP32,
                                                .input  = {.config = {.layout = NWGC}},
                                                .weight = {.config = {.layout = GKXC}},
                                                .output = {.config = {.layout = NWGK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NWGK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKXC_NGKW)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 1,
                                                .direction              = FORWARD,
                                                .data_type              = FP16,
                                                .accumulation_data_type = FP32,
                                                .input  = {.config = {.layout = NGCW}},
                                                .weight = {.config = {.layout = GKXC}},
                                                .output = {.config = {.layout = NGKW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_GNWC_GKXC_GNWK)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 1,
                                                .direction              = FORWARD,
                                                .data_type              = FP16,
                                                .accumulation_data_type = FP32,
                                                .input  = {.config = {.layout = GNWC}},
                                                .weight = {.config = {.layout = GKXC}},
                                                .output = {.config = {.layout = GNWK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNWK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKCX_NGKW)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 1,
                                                .direction              = FORWARD,
                                                .data_type              = FP16,
                                                .accumulation_data_type = FP32,
                                                .input  = {.config = {.layout = NGCW}},
                                                .weight = {.config = {.layout = GKCX}},
                                                .output = {.config = {.layout = NGKW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKYXC_NGKHW)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 2,
                                                .direction              = FORWARD,
                                                .data_type              = FP16,
                                                .accumulation_data_type = FP32,
                                                .input  = {.config = {.layout = NGCHW}},
                                                .weight = {.config = {.layout = GKYXC}},
                                                .output = {.config = {.layout = NGKHW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NHWGC_GKYXC_NHWGK)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 2,
                                                .direction              = FORWARD,
                                                .data_type              = DataType::FP16,
                                                .accumulation_data_type = DataType::FP32,
                                                .input  = {.config = {.layout = NHWGC}},
                                                .weight = {.config = {.layout = GKYXC}},
                                                .output = {.config = {.layout = NHWGK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NHWGK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_GNHWC_GKYXC_GNHWK)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 2,
                                                .direction              = FORWARD,
                                                .data_type              = DataType::FP16,
                                                .accumulation_data_type = DataType::FP32,
                                                .input  = {.config = {.layout = GNHWC}},
                                                .weight = {.config = {.layout = GKYXC}},
                                                .output = {.config = {.layout = GNHWK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNHWK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKCYX_NGKHW)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 2,
                                                .direction              = FORWARD,
                                                .data_type              = DataType::FP16,
                                                .accumulation_data_type = DataType::FP32,
                                                .input  = {.config = {.layout = NGCHW}},
                                                .weight = {.config = {.layout = GKCYX}},
                                                .output = {.config = {.layout = NGKHW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NGCDHW_GKCZYX_NGKDHW)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 3,
                                                .direction              = FORWARD,
                                                .data_type              = DataType::FP16,
                                                .accumulation_data_type = DataType::FP32,
                                                .input  = {.config = {.layout = NGCDHW}},
                                                .weight = {.config = {.layout = GKCZYX}},
                                                .output = {.config = {.layout = NGKDHW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCDHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCZYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKDHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NDHWGC_GKZYXC_NDHWGK)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 3,
                                                .direction              = FORWARD,
                                                .data_type              = DataType::FP16,
                                                .accumulation_data_type = DataType::FP32,
                                                .input  = {.config = {.layout = NDHWGC}},
                                                .weight = {.config = {.layout = GKZYXC}},
                                                .output = {.config = {.layout = NDHWGK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NDHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NDHWGK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_GNDHWC_GKZYXC_GNDHWK)
{
    using namespace enums;
    static constexpr auto sig = ConvSignature<>{.spatial_dim            = 3,
                                                .direction              = FORWARD,
                                                .data_type              = DataType::FP16,
                                                .accumulation_data_type = DataType::FP32,
                                                .input  = {.config = {.layout = GNDHWC}},
                                                .weight = {.config = {.layout = GKZYXC}},
                                                .output = {.config = {.layout = GNDHWK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNDHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNDHWK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(AuxiliaryTensorLayout, AssignsLayoutForG_K_strided)
{
    using CKLayout = LayoutToCK<TensorLayout::G_K_strided>::type;
    EXPECT_TRUE((std::is_same_v<CKLayout, ck::tensor_layout::convolution::G_K>));
}

TEST(AuxiliaryTensorLayout, AssignsLayoutForGC)
{
    using CKLayout = LayoutToCK<TensorLayout::GC>::type;
    EXPECT_TRUE((std::is_same_v<CKLayout, ck::tensor_layout::convolution::GC>));
}

TEST(AuxiliaryTensorLayout, AssignsLayoutForG_C_strided)
{
    using CKLayout = LayoutToCK<TensorLayout::G_C_strided>::type;
    EXPECT_TRUE((std::is_same_v<CKLayout, ck::tensor_layout::convolution::G_C>));
}

TEST(AuxiliaryTensorLayout, EmptyAuxiliaryTensorLayoutIsEmptyTuple)
{
    using ::ck_tile::builder::factory::internal::EmptyAuxiliaryTensorLayout;
    using EmptyLayout = EmptyAuxiliaryTensorLayout::type;
    EXPECT_TRUE((std::is_same_v<EmptyLayout, ck::Tuple<>>));
}

struct MockAuxiliaryTensorConfig
{
    TensorLayout layout;
};

TEST(AuxiliaryTensorLayoutIntegration, SingleBiasTensorWithG_K_Layout)
{
    using namespace enums;

    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = G_K_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, SingleBiasTensorWithGC_Layout)
{
    using namespace enums;

    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::GC}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, SingleBiasTensorWithG_C_Layout)
{
    using namespace enums;

    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = G_C_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_C>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, TwoAuxiliaryTensors)
{
    using namespace enums;

    static constexpr std::array<MockAuxiliaryTensorConfig, 2> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::G_K_strided},
        MockAuxiliaryTensorConfig{.layout = GC}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 2);
    using ExpectedType =
        ck::Tuple<ck::tensor_layout::convolution::G_K, ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, ThreeAuxiliaryTensors)
{
    using namespace enums;

    static constexpr std::array<MockAuxiliaryTensorConfig, 3> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = G_K_strided},
        MockAuxiliaryTensorConfig{.layout = GC},
        MockAuxiliaryTensorConfig{.layout = G_C_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 3);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_K,
                                   ck::tensor_layout::convolution::GC,
                                   ck::tensor_layout::convolution::G_C>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, WorksWith1DConvolution)
{
    using namespace enums;

    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = G_K_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 1, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, WorksWith3DConvolution)
{
    using namespace enums;

    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = GC}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 3, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv2DWithSingleBiasG_K)
{
    using namespace enums;
    using OutputOp = TensorOperation<TensorConfig{.layout = G_K_strided}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 2,
            .direction              = FORWARD,
            .data_type              = DataType::FP16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = NGCHW}},
            .weight                 = {.config = {.layout = GKYXC}},
            .output                 = {.config = {.layout = NGKHW},
                                       .operation =
                                           OutputOp{.elementwise_operation = ElementwiseOperation::SCALE}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv2DWithSingleBiasGC)
{
    using namespace enums;
    using OutputOp = TensorOperation<TensorConfig{.layout = GC}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 2,
            .direction              = FORWARD,
            .data_type              = DataType::BF16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = NHWGC}},
            .weight                 = {.config = {.layout = GKYXC}},
            .output                 = {.config = {.layout = NHWGK},
                                       .operation =
                                           OutputOp{.elementwise_operation = ElementwiseOperation::SCALE}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NHWGK>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv2DWithTwoAuxiliaryTensors)
{
    using namespace enums;
    using OutputOp =
        TensorOperation<TensorConfig{.layout = G_K_strided}, TensorConfig{.layout = GC}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 2,
            .direction              = FORWARD,
            .data_type              = DataType::FP16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = GNHWC}},
            .weight                 = {.config = {.layout = GKYXC}},
            .output                 = {.config    = {.layout = GNHWK},
                                       .operation = OutputOp{.elementwise_operation =
                                                 ElementwiseOperation::SCALEADD_SCALEADD_RELU}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNHWK>));

    using ExpectedDsLayout =
        ck::Tuple<ck::tensor_layout::convolution::G_K, ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv1DWithBias)
{
    using namespace enums;
    using OutputOp = TensorOperation<TensorConfig{.layout = G_K_strided}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 1,
            .direction              = FORWARD,
            .data_type              = DataType::FP32,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = NWGC}},
            .weight                 = {.config = {.layout = GKXC}},
            .output                 = {.config = {.layout = NWGK},
                                       .operation =
                                           OutputOp{.elementwise_operation = ElementwiseOperation::SCALE}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NWGK>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv3DWithBias)
{
    using namespace enums;
    using OutputOp = TensorOperation<TensorConfig{.layout = G_C_strided}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 3,
            .direction              = FORWARD,
            .data_type              = DataType::FP16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = NDHWGC}},
            .weight                 = {.config = {.layout = GKZYXC}},
            .output                 = {.config    = {.layout = NDHWGK},
                                       .operation = OutputOp{.elementwise_operation =
                                                 ElementwiseOperation::BIAS_BNORM_CLAMP}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NDHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NDHWGK>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::G_C>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

} // namespace
