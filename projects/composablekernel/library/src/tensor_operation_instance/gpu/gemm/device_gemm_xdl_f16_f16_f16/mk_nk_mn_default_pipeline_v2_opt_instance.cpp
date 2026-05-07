// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Compilation parameters for a[m, k] * b[n, k] = c[m, n]
using Instances = std::tuple<
// clang-format off
#if CK_EXPERIMENTAL_PIPELINE_V2_INSTANCES        
        // pipeline v2, 1 wave
        //###########| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer| NumPrefetch|           LoopScheduler|             Pipeline|
        //###########|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|            |                        |                     |
        //###########|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|            |                        |                     |
        //###########|      |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |            |                        |                     |
        DeviceGemmXdl<   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   256,    64,   128,     8,  8,   32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,               7,               1,           1,  LoopScheduler::Default,  PipelineVersion::v2>
#endif
    // clang-format on
    >;

void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_default_pipeline_v2_opt_instances(
    OwnerList<InstanceTN>& instances)
{
    add_device_operation_instances(instances, Instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
