// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_splitk_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using ADataType = F8;
using BDataType = F16;
using EDataType = F16;

template <device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched,
          BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp>
using device_grouped_gemm_wmma_universal_f8_f16_f16_mk_kn_mn_instances =
    std::tuple<
        // clang-format off
        //##############################| ALayout| BLayout| DsLayout| ELayout|     AData|     BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //##############################|        |        |         |        |      Type|      Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|              _MBlock_MRepeat| ScalarPerVector|
        //##############################|        |        |         |        |          |          |            |                 |           |          |   Operation|   Operation|    Operation|               |    Stage|      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|              _NBlock_NRepeat|        _NRepeat|
        //##############################|        |        |         |        |          |          |            |                 |           |          |            |            |             |               |         |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Row, DsLayout, ELayout, ADataType, BDataType, AccDataType,        EDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   8,   8,   16,   16,       2,       4,     S<8, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Row, DsLayout, ELayout, ADataType, BDataType, AccDataType,        EDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    64,   2,   2,   16,   16,       2,       4,     S<8, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              2,              2,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              2,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>,
        DeviceGroupedGemm_Wmma_CShuffleV3<    Row,     Row, DsLayout, ELayout, ADataType, BDataType, AccDataType,        EDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    32,   8,   8,   16,   16,       2,       4,     S<4, 32, 1>,     S<2, 0, 1>,     S<2, 0, 1>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 64, 1, 4>,               8, BlkGemmPipeSched, BlkGemmPipelineVer>
        // clang-format on
        >;

void add_device_grouped_gemm_wmma_universal_f8_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Row,
                                                  DsLayout,
                                                  Row,
                                                  ADataType,
                                                  BDataType,
                                                  DsDataType,
                                                  EDataType,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances)
{

    add_device_grouped_gemm_wmma_universal_instances<
        Row,
        Row,
        device_grouped_gemm_wmma_universal_f8_f16_f16_mk_kn_mn_instances>(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
