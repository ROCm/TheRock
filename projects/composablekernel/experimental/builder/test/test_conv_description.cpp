// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/builder/reflect/conv_description.hpp"
#include "ck_tile/builder/reflect/conv_describe.hpp"
#include "testing_utils.hpp"
#include "impl/conv_signature_types.hpp"
#include "impl/conv_algorithm_types.hpp"
#include "ck_tile/builder/conv_signature_utils.hpp"

namespace {

namespace ckb = ck_tile::builder;
namespace ckr = ck_tile::reflect;
namespace ckt = ck_tile::test;

struct TensorOp
{
    ckb::ElementwiseOperation elementwise_operation{ckb::ElementwiseOperation::PASS_THROUGH};
};

struct InvalidTensorOp
{
    int elementwise_operation = 7; // invalid value
};
static_assert(!ckb::TensorOperatorDescriptor<InvalidTensorOp>);

struct TensorConfig
{
    ckb::TensorLayout layout;
    ckb::DataType data_type{ckb::DataType::UNDEFINED_DATA_TYPE};
    ckb::DataType compute_type{ckb::DataType::UNDEFINED_DATA_TYPE};
};

struct TensorConfigNoDataType
{
    ckb::TensorLayout layout;
    ckb::DataType compute_type{ckb::DataType::UNDEFINED_DATA_TYPE};
};

struct ConvTensorNoDataType
{
    TensorConfigNoDataType config;
    TensorOp operation{};
};

struct ConvTensorSimple
{
    TensorConfig config;
};

struct ConvTensorWithOp
{
    TensorConfig config;
    TensorOp operation{};
};

struct ConvTensorWithInvalidOp
{
    TensorConfig config;
    InvalidTensorOp operation{};
};

// Defines the signature of the convolution operation to be tested.
// This includes dimensionality, direction, data layout, and data type.
struct ConvSignature
{
    using enum ckb::DataType;
    using enum ckb::TensorLayout;

    int spatial_dim                      = 2;
    ckb::DataType data_type              = FP16;
    ckb::DataType accumulation_data_type = FP32;
    ConvTensorSimple input               = {.config = {GNHWC}};
    ConvTensorSimple weight              = {.config = {GKYXC}};
    ConvTensorSimple output              = {.config = {GNHWK}};
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignature>);

// Compile time tests for concepts
struct ConvSignatureWithOptionalParams
{
    using enum ckb::DataType;
    using enum ckb::TensorLayout;
    using enum ckb::ConvDirection;
    using enum ckb::ElementwiseOperation;

    int spatial_dim                      = 2;
    ckb::DataType data_type              = FP16;
    ckb::DataType accumulation_data_type = FP32;
    ckb::ConvDirection direction         = FORWARD;
    ConvTensorWithOp input               = {
                      .config = {GNHWC, FP16},
    };
    ConvTensorWithOp weight = {.config = {GKYXC, FP16}};
    ConvTensorWithOp output = {.config = {GNHWK, FP16}, .operation = {SCALE}};
};
static_assert(ckb::ConvSignatureDescriptor<ConvSignatureWithOptionalParams>);

struct ConvSignatureWithInvalidOptionalParams
{
    using enum ckb::DataType;
    using enum ckb::TensorLayout;

    int spatial_dim                      = 2;
    ckb::DataType data_type              = FP16;
    ckb::DataType accumulation_data_type = FP32;
    ConvTensorWithInvalidOp input        = {.config = {GNHWC}};
    ConvTensorWithInvalidOp weight       = {.config = {GKYXC}};
    ConvTensorWithInvalidOp output       = {.config = {GNHWK}};
};
static_assert(!ckb::ConvSignatureDescriptor<ConvSignatureWithInvalidOptionalParams>);

struct DefaultAlgorithm
{
    ckb::test::ThreadBlock thread_block{.block_size = 256,
                                        .tile_size  = {.m = 256, .n = 256, .k = 32}};

    ckb::test::GridwiseXdlGemm gridwise_gemm{.ak1            = 8,
                                             .bk1            = 8,
                                             .m_per_xdl      = 16,
                                             .n_per_xdl      = 16,
                                             .m_xdl_per_wave = 8,
                                             .n_xdl_per_wave = 8};

    ckb::test::TransferABC transfer{
        .a =
            {
                .block_transfer              = {.k0 = 1, .m_n = 128, .k1 = 2},
                .lds_transfer                = {.src_vector_dim            = 2,
                                                .src_scalar_per_vector     = 2,
                                                .lds_dst_scalar_per_vector = 2,
                                                .is_direct_load            = false,
                                                .lds_padding               = false},
                .block_transfer_access_order = {.order = {0, 1, 2}},
                .src_access_order            = {.order = {0, 1, 2}},

            },
        .b =
            {
                .block_transfer              = {.k0 = 1, .m_n = 128, .k1 = 2},
                .lds_transfer                = {.src_vector_dim            = 2,
                                                .src_scalar_per_vector     = 2,
                                                .lds_dst_scalar_per_vector = 2,
                                                .is_direct_load            = false,
                                                .lds_padding               = false},
                .block_transfer_access_order = {.order = {0, 1, 2}},
                .src_access_order            = {.order = {0, 1, 2}},
            },
        .c =
            {
                .thread_cluster_dims =
                    {.m_block = 1, .m_wave_per_xdl = 32, .n_block = 1, .n_wave_per_xdl = 8},
                .epilogue = {.m_xdl_per_wave_per_shuffle = 1,
                             .n_per_wave_per_shuffle     = 1,
                             .scalar_per_vector          = 2},
            },
    };

    ckb::ConvFwdSpecialization fwd_specialization = ckb::ConvFwdSpecialization::DEFAULT;
    ckb::GemmSpecialization gemm_specialization   = ckb::GemmSpecialization::Default;
    ckb::test::BlockGemm block_gemm{.pipeline_version = ckb::PipelineVersion::V4,
                                    .scheduler        = ckb::PipelineScheduler::INTRAWAVE};
};
static_assert(ckb::ConvAlgorithmDescriptor<DefaultAlgorithm>);

struct ConvSignatureUtilsTest1
{
    using enum ckb::DataType;
    using enum ckb::TensorLayout;
    using enum ckb::ConvDirection;
    using enum ckb::ElementwiseOperation;

    int spatial_dim                      = 2;
    ckb::DataType data_type              = FP16;
    ckb::DataType accumulation_data_type = FP32;
    ckb::ConvDirection direction         = FORWARD;
    ConvTensorWithOp input               = {
                      .config = {GNHWC, FP16},
    };
    ConvTensorWithOp weight = {.config = {GKYXC, FP16}};
    ConvTensorWithOp output = {.config = {GNHWK, UNDEFINED_DATA_TYPE}, .operation = {SCALE}};
};

static_assert(ckb::ConvSignatureDescriptor<ConvSignatureUtilsTest1>);

struct ConvSignatureUtilsTest2
{
    using enum ckb::DataType;
    using enum ckb::TensorLayout;
    using enum ckb::ConvDirection;
    using enum ckb::ElementwiseOperation;

    int spatial_dim                                 = 2;
    ckb::DataType data_type                         = FP16;
    ckb::ElementwiseOperation elementwise_operation = CONV_INVSCALE;
    ckb::DataType accumulation_data_type            = FP32;
    ckb::ConvDirection direction                    = FORWARD;
    ConvTensorSimple input                          = {
                                 .config = {GNHWC, FP16},
    };
    ConvTensorNoDataType weight = {.config = {GKYXC}, .operation = {POWER}};
    ConvTensorWithOp output     = {.config = {GNHWK, BF16}, .operation = {GELU}};
};

static_assert(ckb::ConvSignatureDescriptor<ConvSignatureUtilsTest2>);

TEST(ConvUtilsTest, getDataType1)
{
    using enum ckb::DataType;
    static constexpr const ConvSignatureUtilsTest1 SIGNATURE;
    EXPECT_THAT(ckb::getInputDataType<SIGNATURE>(), FP16);
    EXPECT_THAT(ckb::getWeightDataType<SIGNATURE>(), FP16);
    EXPECT_THAT(ckb::getOutputDataType<SIGNATURE>(), FP16);
    EXPECT_THAT(ckb::getDataTypeIfCommon<SIGNATURE>(), FP16);
}

TEST(ConvUtilsTest, getDataType2)
{
    using enum ckb::DataType;
    static constexpr const ConvSignatureUtilsTest2 SIGNATURE;
    EXPECT_THAT(ckb::getInputDataType<SIGNATURE>(), FP16);
    EXPECT_THAT(ckb::getWeightDataType<SIGNATURE>(), FP16);
    EXPECT_THAT(ckb::getOutputDataType<SIGNATURE>(), BF16);
    EXPECT_THAT(ckb::getDataTypeIfCommon<SIGNATURE>(), UNDEFINED_DATA_TYPE);
}

TEST(ConvUtilsTest, getElementwiseOperation1)
{
    using enum ckb::ElementwiseOperation;
    static constexpr const ConvSignatureUtilsTest1 SIGNATURE;
    EXPECT_THAT(ckb::getInputElementwiseOperation<SIGNATURE>(), PASS_THROUGH);
    EXPECT_THAT(ckb::getWeightElementwiseOperation<SIGNATURE>(), PASS_THROUGH);
    EXPECT_THAT(ckb::getOutputElementwiseOperation<SIGNATURE>(), SCALE);
}

TEST(ConvUtilsTest, getElementwiseOperation2)
{
    using enum ckb::ElementwiseOperation;
    static constexpr const ConvSignatureUtilsTest2 SIGNATURE;
    EXPECT_THAT(ckb::getInputElementwiseOperation<SIGNATURE>(), CONV_INVSCALE);
    EXPECT_THAT(ckb::getWeightElementwiseOperation<SIGNATURE>(), POWER);
    EXPECT_THAT(ckb::getOutputElementwiseOperation<SIGNATURE>(), GELU);
}

TEST(ConvDescriptionTest, DefaultInstanceHasBriefDescription)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Instance = ckb::ConvBuilder<SIGNATURE, ALGORITHM>::Instance;
    EXPECT_THAT(ckr::describe<Instance>().brief(), ckt::StringEqWithDiff("2D Forward convolution"));
}

TEST(ConvDescriptionTest, DefaultInstanceHasDetailedDescription)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Instance = ckb::ConvBuilder<SIGNATURE, ALGORITHM>::Instance;
    EXPECT_THAT(ckr::describe<Instance>().detailed(),
                ckt::StringEqWithDiff( //
                    "2D Forward Convolution Kernel\n"
                    "├─ Signature\n"
                    "│  ├─ Tensor Type: FP16\n"
                    "│  ├─ Input Layout: GNHWC\n"
                    "│  ├─ Weight Layout: GKYXC\n"
                    "│  ├─ Output Layout: GNHWK\n"
                    "│  ├─ Input elementwise operation: PASS_THROUGH\n"
                    "│  ├─ Weights elementwise operation: PASS_THROUGH\n"
                    "│  └─ Output elementwise operation: PASS_THROUGH\n"
                    "└─ Algorithm\n"
                    "   ├─ Thread block size: 256\n"
                    "   ├─ Data tile size: 256×256×32\n"
                    "   ├─ Gemm padding: DEFAULT\n"
                    "   ├─ Convolution specialization: DEFAULT\n"
                    "   ├─ Pipeline version: V4\n"
                    "   ├─ Pipeline scheduler: INTRAWAVE\n"
                    "   ├─ Warp Gemm parameters: \n"
                    "   │  ├─ subtile size: 16×16\n"
                    "   │  └─ Number of warp gemm iterations: 8×8\n"
                    "   └─ Memory access:\n"
                    "      ├─ A Tile transfer: \n"
                    "      │  ├─ Tile dimensions: 4×256×8×\n"
                    "      │  ├─ The innermost K subdimension size: 8\n"
                    "      │  ├─ Spatial thread distribution over the data tile: 0×1×2\n"
                    "      │  ├─ The order of accessing data tile axes: 0×1×2\n"
                    "      │  ├─ Vectorized memory access axis index (with contiguous memory): 2\n"
                    "      │  ├─ Vector access (GMEM read) instruction size: 2\n"
                    "      │  ├─ Vector access (LDS write) instruction size: 2\n"
                    "      │  └─ LDS data layout padding (to prevent bank conflicts): 2\n"
                    "      ├─ B Tile transfer: \n"
                    "      │  ├─ Tile dimensions: 4×256×8×\n"
                    "      │  ├─ The innermost K subdimension size: 8\n"
                    "      │  ├─ Spatial thread distribution over the data tile: 0×1×2\n"
                    "      │  ├─ The order of accessing data tile axes: 0×1×2\n"
                    "      │  ├─ Vectorized memory access axis index (with contiguous memory): 2\n"
                    "      │  ├─ Vector access (GMEM read) instruction size: 2\n"
                    "      │  ├─ Vector access (LDS write) instruction size: 2\n"
                    "      │  └─ LDS data layout padding (to prevent bank conflicts): 2\n"
                    "      └─ C Tile transfer: \n"
                    "         ├─ Data shuffle (number of gemm instructions per iteration): 1×1\n"
                    "         ├─ Spatial thread distribution used to store data: 1×32×1×8\n"
                    "         └─ Vector access (GMEM write) instruction size: 2"));
}

TEST(ConvDescriptionTest, DefaultInstanceHasInstanceString)
{
    static constexpr const ConvSignature SIGNATURE;
    static constexpr const DefaultAlgorithm ALGORITHM;
    using Instance = ckb::ConvBuilder<SIGNATURE, ALGORITHM>::Instance;

    // Get the instance string from the description
    std::string instance_str = ckr::describe<Instance>().instance_string();

    // Verify that the instance string is not empty
    EXPECT_FALSE(instance_str.empty());

    // Verify that it contains the device operation name
    // The exact format depends on the InstanceTraits implementation
    EXPECT_THAT(instance_str, ::testing::HasSubstr("DeviceGroupedConvFwdMultipleABD"));
}

// NOTE: BackwardDataInstanceHasDetailedDescription test is disabled because ConvFactory
// does not have a specialization for backward data convolutions. The test fails with:
//   "implicit instantiation of undefined template 'ck_tile::builder::ConvFactory<...>'"
//
// To enable this test, a ConvFactory specialization for backward data operations must be
// implemented first.
//
// TEST(ConvDescriptionTest, BackwardDataInstanceHasDetailedDescription)
// {
//     struct BackwardDataSignature
//     {
//         int spatial_dim              = 2;
//         ckb::ConvDirection direction = ckb::ConvDirection::BACKWARD_DATA;
//         ckb::GroupConvLayout layout  = ckb::GroupConvLayout2D::GNHWC_GKYXC_GNHWK;
//         ckb::DataType data_type      = ckb::DataType::FP16;
//         ckb::ElementwiseOperation elementwise_operation =
//         ckb::ElementwiseOperation::PASS_THROUGH; ckb::GroupConvDeviceOp device_operation =
//             ckb::BwdDataGroupConvDeviceOperation::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1;
//     };
//     static_assert(ckb::ConvSignatureDescriptor<BackwardDataSignature>);
//
//     static constexpr const BackwardDataSignature SIGNATURE;
//     static constexpr const DefaultAlgorithm ALGORITHM;
//     using Builder = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
//
//     // Verify Brief works
//     EXPECT_THAT(ckr::Describe<Builder>().brief(),
//                 ckt::StringEqWithDiff("2D Backward Data convolution"));
//
//     // Verify detailed works - to be updated once ConvFactory is implemented
//     EXPECT_THAT(ckr::Describe<Builder>().detailed(),
//                 ckt::StringEqWithDiff("PLACEHOLDER"));
// }

} // namespace
