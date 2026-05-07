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
#if CK_EXPERIMENTAL_INTER_WAVE_INSTANCES        
        // pipeline v1, 2 waves
        //###########| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer| NumPrefetch|          LoopScheduler|                     Pipeline|
        //###########|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|            |                       |                             |
        //###########|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|            |                       |                             |
        //###########|      |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |            |                       |                             |
        DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmMNPadding,   64,    16,    16,     4,  8,   16,   16,    1,    1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,      true,               7,               1,           1,  LoopScheduler::Interwave,        PipelineVersion::v1>
#endif
    // clang-format on
    >;

void add_device_gemm_xdl_f16_f16_f16_km_nk_mn_irregular_interwave_pipeline_v1_instances(
    OwnerList<InstanceNN>& instances)
{
    add_device_operation_instances(instances, Instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
