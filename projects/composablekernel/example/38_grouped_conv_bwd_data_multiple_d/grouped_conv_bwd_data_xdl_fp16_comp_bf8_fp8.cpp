// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"
#include "common.hpp"

using OutDataType      = FP16;
using WeiDataType      = FP16;
using AccDataType      = FP32;
using CShuffleDataType = FP16;
using DsDataType       = ck::Tuple<>;
using InDataType       = FP16;
using AComputeType     = BF8;
using BComputeType     = FP8;

using OutLayout = ck::tensor_layout::convolution::GNHWK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using DsLayout  = ck::Tuple<>;
using InLayout  = ck::tensor_layout::convolution::GNHWC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = PassThrough;

static constexpr auto LoopSched = ck::make_default_loop_scheduler();

// clang-format off
using DeviceConvInstance = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
// ######| NDimSpatial|   ALayout|   BLayout|   DsLayout|  ELayout|       AData|       BData|     AccData|         CShuffle|       DsData|      EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer|    MXdl|    NXdl|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffleMXdl| CShuffleNXdl|   CDEBlockTransfer| CDEBlockTransfer|      Loop|     ACompute|     BCompute|
// ######|            |          |          |           |         |        Type|        Type|        Type|         DataType|         Type|       Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN| PrefetchStage|  Size| Block| Block| Block|    |    |  XDL|  XDL| PerWave| PerWave|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      PerWave|      PerWave|  _MBlock_MPerBlock|  ScalarPerVector| Scheduler|         Type|         Type|
// ######|            |          |          |           |         |            |            |            |                 |             |           |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|       _NPerBlock|          |             |             |
// ######|            |          |          |           |         |            |            |            |                 |             |           |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |          |             |             |
         < NDimSpatial, OutLayout, WeiLayout,   DsLayout, InLayout, OutDataType, WeiDataType, AccDataType, CShuffleDataType,   DsDataType, InDataType, OutElementOp, WeiElementOp,    InElementOp,  ConvBwdDataDefault,  true,  true,             1,   256,   128,   256,    32,   8,   2,   16,   16,       4,       8,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,            1,            1,     S<1, 32, 1, 8>,                4, LoopSched, AComputeType, BComputeType>;
// clang-format on

#include "run_grouped_conv_bwd_data_example.inc"

int main(int argc, char* argv[])
{
    // temp disable on gfx11
    if(ck::is_gfx11_supported())
    {
        return 0;
    }
    return run_grouped_conv_bwd_data_example(argc, argv);
}
