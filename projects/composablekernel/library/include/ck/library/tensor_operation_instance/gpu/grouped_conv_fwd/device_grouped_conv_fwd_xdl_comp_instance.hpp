// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP8
using F8 = ck::f8_t;
#endif

#ifdef CK_ENABLE_BF8
using BF8 = ck::bf8_t;
#endif

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using TF32 = ck::tf32_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Empty_Tuple = ck::Tuple<>;

using namespace ck::tensor_layout::convolution;

using PassThrough               = ck::tensor_operation::element_wise::PassThrough;
using BiasNormalizeInInferClamp = ck::tensor_operation::element_wise::BiasNormalizeInInferClamp;
using AddClamp                  = ck::tensor_operation::element_wise::AddClamp;
using Clamp                     = ck::tensor_operation::element_wise::Clamp;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd1x1P0 = ConvolutionForwardSpecialization::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 = ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

static constexpr auto ConvFwdOddC =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

// double rate mfma instances on gfx950
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_bf16_comp_instances_2x = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,  16,  16,  32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          1,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          1,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_bf16_comp_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // Compute friendly
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    32,   8,   8,  32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   256,    32,   8,   8,  32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   128,    32,   8,   8,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,    64,    64,   8,   8,  32,   32,    2,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,    64,   128,    64,   8,   8,  32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,    64,    64,    64,   8,   8,  32,   32,    1,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>
    // clang-format on
    >;

// instances not working on gfx950
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_bf16_comp_instances_part2 = std::tuple<
    // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  16,   16,    8,    8,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           2,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        // AGPR Spill when use permuted lds layout. so, use padding for these two.
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   224,   256,    64,   8,   8,  16,   16,    7,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           2,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   224,    64,   8,   8,  16,   16,    8,    7,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          2,           1,                   S<1, 64, 1, 4>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>
    // clang-format on
    >;

// double rate mfma instances on gfx950
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_f16_comp_instances_2x = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,  16,  16,  32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          1,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          1,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_f16_comp_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>
    // clang-format on
    >;

// instances not working on gfx950
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_f16_comp_instances_part2 = std::tuple<
    // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    32,   8,   8,  32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  16,   16,    8,    8,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           2,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        // AGPR Spill when use permuted lds layout. so, use padding for these two.
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   224,   256,    64,   8,   8,  16,   16,    7,    8,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           2,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   224,    64,   8,   8,  16,   16,    8,    7,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          2,           1,                   S<1, 64, 1, 4>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   256,    32,   8,   8,  32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   128,    32,   8,   8,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_f32_comp_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    32,   8,   8,  32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_f32_tf32_comp_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    32,   8,   8,  32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4, TF32, TF32>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3, TF32, TF32>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5, TF32, TF32>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F32,   F32,     F32,      F32,    DsDataTypes,   F32, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              4,              4,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1, TF32, TF32>
    // clang-format on
    >;

// double rate mfma instances on gfx950
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_int8_comp_instances_2x = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,DsLayout,ELayout,int8_t,int8_t,int32_t,   int8_t,    DsLayout,int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,             256,   128,   128,   128,  32,  32,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,              16,        1,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,         1,            1,           1,              S<1, 64, 1, 4>,              16,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_int8_comp_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    32,   8,   8,  32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>
    // clang-format on
    >;

// instances not working on gfx950
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_int8_comp_instances_part2 = std::tuple<
    // clang-format off
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v4>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  32,   32,    4,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v5>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   256,    32,   8,   8,  16,   16,    8,    8,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           2,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        // AGPR Spill when use permuted lds layout. so, use padding for these two.
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   256,    32,   8,   8,  32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   256,   128,    32,   8,   8,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,   8,   8,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,              8,              8,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlockGemmPipelineScheduler::Interwave, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
