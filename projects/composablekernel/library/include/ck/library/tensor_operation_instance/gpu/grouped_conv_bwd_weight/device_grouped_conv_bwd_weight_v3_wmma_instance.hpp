// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

#ifdef CK_ENABLE_FP8
using F8 = ck::f8_t;
#endif

#ifdef CK_ENABLE_BF8
using BF8 = ck::bf8_t;
#endif

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

static constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler     = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion PipelineVersion = BlockGemmPipelineVersion::v1>
using device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_f16_instances = std::tuple<
    // clang-format off
    //#########################################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|   ConvBackward| Block|  MPer|  NPer|  KPer| ABK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CShuffleBlockTransfer|  CShuffleBlockTransfer| BlockGemm|       BlockGemm|
    //#########################################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|         Weight|  Size| Block| Block| Block|     | Wmma| Wmma|        |        |     ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|        ClusterLengths|        ScalarPerVector|  Pipeline|        Pipeline|
    //#########################################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation| Specialization|      |      |      |      |     |     |     |        |        | Lengths_AK0_M_AK1|     ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |              |      PerVector|  PerVector_BK1|          |  PerShuffle|  PerShuffle|      MBlock_MPerBlock|             _NPerBlock| Scheduler|         Version|
    //#########################################|           |         |          |          |       |        |        |        |            |            |            |               |      |      |      |      |     |     |     |        |        |                  |                 |               |               |               |               |          |                  |               |               |              |               |               |          |            |            |      NBlock_NPerBlock|                       |          |                |
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    32,    32,    32,    8,   16,   16,       2,       1,        S<4, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              2,              2,         0,       S<4, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              2,              2,         0,           1,           1,         S<1, 8, 1, 8>,                      2, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         0,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    64,    64,    64,    8,   16,   16,       4,       2,        S<8, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,        S<8, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 4>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,   128,   256,    64,    8,   16,   16,       8,       2,       S<8, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<8, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    48,    64,   128,    8,   16,   16,       3,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    96,   128,    64,    8,   16,   16,       6,       2,       S<8, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<8, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    64,    64,   128,    8,   16,   16,       4,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,    96,   128,   128,    8,   16,   16,       6,       1,      S<16, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,      S<16, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>
    // DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         0,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>, // Incorrect results for at least GemmDefault
    // DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>  // Incorrect results for at least GemmDefault
    // clang-format on
    >;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler     = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion PipelineVersion = BlockGemmPipelineVersion::v1>
using device_grouped_conv_bwd_weight_v3_wmma_c_shuffle_bf16_instances = std::tuple<
    // clang-format off
    //#########################################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|   ConvBackward| Block|  MPer|  NPer|  KPer| ABK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CShuffleBlockTransfer|  CShuffleBlockTransfer| BlockGemm|       BlockGemm|
    //#########################################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|         Weight|  Size| Block| Block| Block|     | Wmma| Wmma|        |        |     ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|        ClusterLengths|        ScalarPerVector|  Pipeline|        Pipeline|
    //#########################################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation| Specialization|      |      |      |      |     |     |     |        |        | Lengths_AK0_M_AK1|     ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |              |      PerVector|  PerVector_BK1|          |  PerShuffle|  PerShuffle|      MBlock_MPerBlock|             _NPerBlock| Scheduler|         Version|
    //#########################################|           |         |          |          |       |        |        |        |            |            |            |               |      |      |      |      |     |     |     |        |        |                  |                 |               |               |               |               |          |                  |               |               |              |               |               |          |            |            |      NBlock_NPerBlock|                       |          |                |
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    32,    32,    32,    8,   16,   16,       2,       1,        S<4, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              2,              2,         0,       S<4, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              2,              2,         0,           1,           1,         S<1, 8, 1, 8>,                      2, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         0,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,   128,   128,    32,    8,   16,   16,       8,       2,       S<4, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<4, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              4,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    64,    64,    64,    64,    8,   16,   16,       4,       2,        S<8, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,        S<8, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 4>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,   128,   256,    64,    8,   16,   16,       8,       2,       S<8, 32, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              4,              8,         1,       S<8, 32, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    48,    64,   128,    8,   16,   16,       3,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    96,   128,    64,    8,   16,   16,       6,       2,       S<8, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<8, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   128,    64,    64,   128,    8,   16,   16,       4,       1,       S<16, 8, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              8,              8,         0,       S<16, 8, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,        S<1, 16, 1, 8>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,   256,    96,   128,   128,    8,   16,   16,       6,       1,      S<16, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,      S<16, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              8,              8,         0,           1,           1,       S<1, 16, 1, 16>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         0,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>,
    DeviceGroupedConvBwdWeight_Wmma_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,       ConvSpec,    96,    96,    96,    48,    8,   16,   16,       6,       2,       S<6, 16, 1>,       S<2, 0, 1>,     S<1, 0, 2>,              1,              6,              8,         1,       S<6, 16, 1>,     S<2, 0, 1>,     S<1, 0, 2>,             1,              6,              8,         0,           1,           1,        S<1, 16, 1, 6>,                      8, Scheduler, PipelineVersion>
    //clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
