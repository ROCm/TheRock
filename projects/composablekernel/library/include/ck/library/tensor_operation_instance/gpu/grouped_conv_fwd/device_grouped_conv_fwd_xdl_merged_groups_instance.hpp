// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

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

static constexpr auto ConvFwd3x3 = ConvolutionForwardSpecialization::Filter3x3;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_merged_groups_bf16_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer| ACompute| BCompute| BlockGemm| NumGroups|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|     Type|     Type|  Pipeline|   ToMerge|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|         |         | Scheduler|          |
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |         |         |          |          |
        // Instances with NumGroupsPerBatch > 1
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, BF16, BF16, LoopScheduler::Default, 8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, BF16, BF16, LoopScheduler::Default, 16>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, BF16, BF16, LoopScheduler::Default, 32>
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
using device_grouped_conv_fwd_xdl_merged_groups_bf16_instances_2x = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer| ACompute| BCompute| BlockGemm| NumGroups|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|     Type|     Type|  Pipeline|   ToMerge|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|         |         | Scheduler|          |
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |         |         |          |          |
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     32,   8, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, BF16, BF16, LoopScheduler::Default, 8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     32,   8, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, BF16, BF16, LoopScheduler::Default, 16>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,  BF16,  BF16,     F32,     BF16,    DsDataTypes,  BF16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     32,   8, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, BF16, BF16, LoopScheduler::Default, 32>
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
using device_grouped_conv_fwd_xdl_merged_groups_f16_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // Instances with NumGroupsPerBatch > 1
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F16, F16, LoopScheduler::Default, 8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F16, F16, LoopScheduler::Default, 16>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F16, F16, LoopScheduler::Default, 32>
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
using device_grouped_conv_fwd_xdl_merged_groups_f16_instances_2x = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // Instances with NumGroupsPerBatch > 1
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     32,   8, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F16, F16, LoopScheduler::Default, 8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     32,   8, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F16, F16, LoopScheduler::Default, 16>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   F16,   F16,     F32,      F16,    DsDataTypes,   F16, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     32,   8, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F16, F16, LoopScheduler::Default, 32>
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
using device_grouped_conv_fwd_xdl_merged_groups_f32_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // Instances with NumGroupsPerBatch > 1
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,    F32,     F32,     F32,     F32, DsDataTypes,   F32,  PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F32, F32, LoopScheduler::Default, 8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,    F32,     F32,     F32,     F32, DsDataTypes,   F32,  PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F32, F32, LoopScheduler::Default, 16>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,    F32,     F32,     F32,     F32, DsDataTypes,   F32,  PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,     16,   4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, F32, F32, LoopScheduler::Default, 32>
    // clang-format on
    >;

#if defined(__gfx950__)
constexpr auto _k_per_block = 32;
#else
constexpr auto _k_per_block = 16;
#endif

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataTypes  = Tuple<>,
          typename OutElementOp = PassThrough>
using device_grouped_conv_fwd_xdl_merged_groups_f32_tf32_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // Instances with NumGroupsPerBatch > 1
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,    F32,     F32,     F32,     F32, DsDataTypes,   F32,  PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,_k_per_block,4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, TF32, TF32, LoopScheduler::Default, 8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,    F32,     F32,     F32,     F32, DsDataTypes,   F32,  PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,_k_per_block,4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, TF32, TF32, LoopScheduler::Default, 16>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,    F32,     F32,     F32,     F32, DsDataTypes,   F32,  PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,    64,    16,_k_per_block,4, 4,  16,   16,    4,    1,  S< 4, 16,  1>, S<0, 2, 1>,     S<0, 2, 1>,                   1,              4,              4,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              4,      1,           1,           1,   S<1, 16, 1, 4>,                  1, TF32, TF32, LoopScheduler::Default, 32>
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
using device_grouped_conv_fwd_xdl_merged_groups_int8_instances = std::tuple<
    // clang-format off
        //########################################|     NumDim|      A|      B|          Ds|      E| AData| BData| AccData| CShuffle|          Ds| EData|           A|           B|         CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|    Spatial| Layout| Layout|      Layout| Layout|  Type|  Type|    Type| DataType|    DataType|  Type| Elementwise| Elementwise| Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |   Operation|   Operation|   Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|           |       |       |            |       |      |      |        |         |            |      |            |            |            |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // Instances with NumGroupsPerBatch > 1
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,     32,    64,     32,   8, 8,  32,   32,    1,    2,  S< 4, 16,  1>, S<1, 0, 2>,     S<1, 0, 2>,                   2,              8,              8,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      1,           1,           1,   S<1, 16, 1, 4>,                  1, int8_t, int8_t, LoopScheduler::Default, 8>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,     32,    64,     32,   8, 8,  32,   32,    1,    2,  S< 4, 16,  1>, S<1, 0, 2>,     S<1, 0, 2>,                   2,              8,              8,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      1,           1,           1,   S<1, 16, 1, 4>,                  1, int8_t, int8_t, LoopScheduler::Default, 16>,
        DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<NDimSpatial,ALayout,BLayout,    DsLayout,ELayout,   int8_t,   int8_t,     int32_t,      int8_t,    DsDataTypes,   int8_t, PassThrough, PassThrough, OutElementOp,                  ConvSpec, GemmMNKPadding,  1,  64,     32,    64,     32,   8, 8,  32,   32,    1,    2,  S< 4, 16,  1>, S<1, 0, 2>,     S<1, 0, 2>,                   2,              8,              8,      1,  S< 4, 16,  1>,   S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      1,           1,           1,   S<1, 16, 1, 4>,                  1, int8_t, int8_t, LoopScheduler::Default, 32>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
