// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp"
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
using TF32 = ck::tf32_t;

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
          BlockGemmPipelineScheduler Scheduler,
          BlockGemmPipelineVersion PipelineVersion>
using device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f32_instances = std::tuple<
    // clang-format off
        //#########################################|     Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|   CBlockTransfer|  CBlockTransfer| BlockGemm| BlockGemm|
        //#########################################|     Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|   ClusterLengths| ScalarPerVector|  Pipeline|  Pipeline|
        //#########################################| Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|     ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| MBlock_MPerBlock|    NWaveNPerXdl| Scheduler|   Version|
        //#########################################|        |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |     |     |     |     |                |                 |               |               |               |               |          |                |               |               |              |               |               |          |            |            | NBlock_NPerBlock|                |          |          |
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    16,    16,     32,   8,   16,   16,    1,    1,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              1,              4,      false,  S<4, 8,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              1,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>
    // clang-format on
    >;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler,
          BlockGemmPipelineVersion PipelineVersion>
using device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f32_tf32_instances = std::tuple<
    // clang-format off
        //#########################################|     Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|   CBlockTransfer|  CBlockTransfer| BlockGemm| BlockGemm|
        //#########################################|     Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|   ClusterLengths| ScalarPerVector|  Pipeline|  Pipeline|
        //#########################################| Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|     ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| MBlock_MPerBlock|    NWaveNPerXdl| Scheduler|   Version|
        //#########################################|        |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |     |     |     |     |                |                 |               |               |               |               |          |                |               |               |              |               |               |          |            |            | NBlock_NPerBlock|                |          |          |
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    16,    16,     32,   8,   16,   16,    1,    1,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              1,              4,      false,  S<4, 8,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              1,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion, TF32, TF32>
    // clang-format on
    >;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler,
          BlockGemmPipelineVersion PipelineVersion>
using device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_f16_instances = std::tuple<
    // clang-format off
        //#########################################|     Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|   CBlockTransfer|  CBlockTransfer| BlockGemm| BlockGemm|
        //#########################################|     Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|   ClusterLengths| ScalarPerVector|  Pipeline|  Pipeline|
        //#########################################| Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|     ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| MBlock_MPerBlock|    NWaveNPerXdl| Scheduler|   Version|
        //#########################################|        |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |     |     |     |     |                |                 |               |               |               |               |          |                |               |               |              |               |               |          |            |            | NBlock_NPerBlock|                |          |          |
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    32,     32,   8,   32,   32,    1,    1,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              2,              2,      false,  S<4, 16, 1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              2,              2,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    64,     32,   8,   32,   32,    1,    2,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              4,              4,      false,  S<4, 16, 1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              4,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,   128,     32,   8,   32,   32,    1,    4,  S<4, 4,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              8,              8,      false,  S<4, 16, 1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              8,              8,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    64,    32,     32,   8,   32,   32,    2,    1,  S<4, 16, 1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              4,              4,      false,  S<4, 8,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              4,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,   128,    32,     32,   8,   32,   32,    4,    1,  S<4, 16, 1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              8,              8,      false,  S<4, 4,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              8,              8,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,       
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    64,    80,     32,   8,   16,   16,    4,    5,  S<4, 16, 1>, S<2, 0, 1>,  S<2, 0, 1>,                   1,              4,              4,      false,  S<4, 16, 1>,  S<2, 0, 1>,  S<2, 0, 1>,                1,              5,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    64,   112,     32,   8,   16,   16,    4,    7,  S<4, 16, 1>, S<2, 0, 1>,  S<2, 0, 1>,                   1,              4,              4,      false,  S<4, 16, 1>,  S<2, 0, 1>,  S<2, 0, 1>,                1,              7,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>
    // clang-format on
    >;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec,
          BlockGemmPipelineScheduler Scheduler,
          BlockGemmPipelineVersion PipelineVersion>
using device_grouped_conv_bwd_weight_v3_xdl_c_shuffle_bf16_instances = std::tuple<
    // clang-format off
        //#########################################|     Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|   ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|   CBlockTransfer|  CBlockTransfer| BlockGemm| BlockGemm|
        //#########################################|     Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|    ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|   ClusterLengths| ScalarPerVector|  Pipeline|  Pipeline|
        //#########################################| Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|     ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| MBlock_MPerBlock|    NWaveNPerXdl| Scheduler|   Version|
        //#########################################|        |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |     |     |     |     |                |                 |               |               |               |               |          |                |               |               |              |               |               |          |            |            | NBlock_NPerBlock|                |          |          |
        // generic instance
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    32,     32,   8,   32,   32,    1,    1,  S<4, 8,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                  1,              2,              2,      false,  S<4, 16,  1>, S<2, 0, 1>,  S<1, 0, 2>,                1,              2,              2,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    64,     32,   8,   32,   32,    1,    2,  S<4, 8,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                  1,              4,              4,      false,  S<4, 16,  1>, S<2, 0, 1>,  S<1, 0, 2>,                1,              4,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,   128,     32,   8,   32,   32,    1,    4,  S<4, 4,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                  1,              8,              8,      false,  S<4, 16,  1>, S<2, 0, 1>,  S<1, 0, 2>,                1,              8,              8,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    64,    32,     32,   8,   32,   32,    2,    1,  S<4, 16, 1>,  S<2, 0, 1>,  S<1, 0, 2>,                  1,              4,              4,      false,  S<4, 8,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              4,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,   128,    32,     32,   8,   32,   32,    4,    1,  S<4, 16, 1>,  S<2, 0, 1>,  S<1, 0, 2>,                  1,              8,              8,      false,  S<4, 4,  1>,  S<2, 0, 1>,  S<1, 0, 2>,                1,              8,              8,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    64,    80,     32,   8,   16,   16,    4,    5,  S<4, 16, 1>,  S<2, 0, 1>,  S<2, 0, 1>,                  1,              4,              4,      false,  S<4, 16,  1>,  S<2, 0, 1>,  S<2, 0, 1>,               1,              5,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>,
        DeviceGroupedConvBwdWeight_Xdl_CShuffleV3< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,    BF16,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    64,   112,     32,   8,   16,   16,    4,    7,  S<4, 16, 1>,  S<2, 0, 1>,  S<2, 0, 1>,                  1,              4,              4,      false,  S<4, 16,  1>,  S<2, 0, 1>,  S<2, 0, 1>,               1,              7,              4,      false,           1,           1,   S<1, 8, 1, 8>,                  2, Scheduler, PipelineVersion>
    //clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
