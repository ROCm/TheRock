// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// irregular tile size
using Instances = std::tuple<
    // clang-format off
        // pipeline v1, 1 wave
        //###########| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer| NumPrefetch|          LoopScheduler|                     Pipeline|
        //###########|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|            |                       |                             |
        //###########|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|            |                       |                             |
        //###########|      |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |            |                       |                             |
        DeviceGemmXdl<   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNPadding,    64,    64,    32,     4,  8,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Default,        PipelineVersion::v1>,
        DeviceGemmXdl<   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNPadding,   256,   128,   144,     8,  8,   16,   16,    2,    9,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<8,  8, 4>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              2,              2,      true,               7,               1,           1,  LoopScheduler::Default,        PipelineVersion::v1>,
        DeviceGemmXdl<   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNPadding,   256,   128,   144,     4,  8,   16,   16,    2,    9,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<4, 16, 4>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              2,              2,      true,               7,               1,           1,  LoopScheduler::Default,        PipelineVersion::v1>
    // clang-format on
    >;

void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_irregular_default_pipeline_v1_instances(
    OwnerList<InstanceTN>& instances)
{
    add_device_operation_instances(instances, Instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
