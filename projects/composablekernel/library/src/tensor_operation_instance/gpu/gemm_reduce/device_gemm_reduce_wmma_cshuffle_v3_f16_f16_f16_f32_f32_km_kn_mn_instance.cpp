// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_reduce_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16              = ck::half_t;
using F32              = float;
using ReducePtrsGlobal = ck::Tuple<F32*, F32*>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ReduceSum   = ck::reduce::Add;
using ReduceOps   = ck::Tuple<ReduceSum, ReduceSum>;

using Div                 = ck::tensor_operation::element_wise::UnaryDivide;
using Identity            = ck::tensor_operation::element_wise::PassThrough;
using Square              = ck::tensor_operation::element_wise::UnarySquare;
using ReduceInElementOps  = ck::Tuple<Identity, Square>;
using ReduceOutElementOps = ck::Tuple<Div, Div>;

using ReduceMemOp = ck::InMemoryDataOperationEnumSequence<ck::InMemoryDataOperationEnum::AtomicAdd,
                                                          ck::InMemoryDataOperationEnum::AtomicAdd>;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

// c[m, n] = a[k, m] * b[k, n]
using device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_kn_mn_instances =
    std::tuple<
        // clang-format off
        //##############################| ALayout| BLayout| ELayout|AData| BData| EData|      Acc| CShuffle| ReduceAcc| ReducePtrsGlobal|           A|           B|           C|    Reduce|           ReduceIn|           ReduceAcc| ReduceGlobal|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CDEShuffleBlockTransferClusterLengths|  CDEShuffleBlockTransfer|              CReduce| CReduceThreadLds2VGprCopy| CReduceThreadVgpr2GlobalCopy|    BlkGemm|                      BlkGemm|
        //##############################|        |        |        | Type|  Type|  Type| DataType| DataType|  DataType|                 | Elementwise| Elementwise| Elementwise| Operation|        Elementwise|         Elementwise|   MemoryData| Specialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|                     _MBlock_MPerBlock|          ScalarPerVector| ThreadClusterLengths|     SrcDstScalarPerVector|        SrcDstScalarPerVector|  PipeSched|                  PipelineVer|
        //##############################|        |        |        |     |      |      |         |         |          |                 |   Operation|   Operation|   Operation|          |         Operations|          Operations|    Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                     _NBlock_NPerBlock|                         | _MPerBlock_NPerBlock|                _NPerBlock|                   _MPerBlock|           |                             |
        //##############################|        |        |        |     |      |      |         |         |          |                 |            |            |            |          |                   |                    |             |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                      |                         |                     |                          |                             |           |                             |
        // v1 Intrawave
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   256,   128,    32,   2,   2,   16,   16,       8,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        8,             S<32, 8>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   128,   256,    32,   2,   2,   16,   16,       2,       8,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        4,             S<64, 4>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   128,   128,   128,    32,   2,   2,   16,   16,       4,       4,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 4>,                        8,             S<32, 4>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   128,   128,    32,   2,   2,   16,   16,       4,       2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        8,             S<32, 8>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   128,    64,   128,    32,   2,   2,   16,   16,       2,       4,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 4>,                        8,             S<32, 4>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v1>,

        // v1 Interwave
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   256,   128,    32,   2,   2,   16,   16,       8,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        8,             S<32, 8>,                         4,                            1, Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   128,   256,    32,   2,   2,   16,   16,       2,       8,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        4,             S<64, 4>,                         4,                            1, Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   128,   128,   128,    32,   2,   2,   16,   16,       4,       4,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 4>,                        8,             S<32, 4>,                         4,                            1, Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   128,   128,    32,   2,   2,   16,   16,       4,       2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        8,             S<32, 8>,                         4,                            1, Interwave, BlockGemmPipelineVersion::v1>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   128,    64,   128,    32,   2,   2,   16,   16,       2,       4,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 4>,                        8,             S<32, 4>,                         4,                            1, Interwave, BlockGemmPipelineVersion::v1>,

        // v3 Intrawave
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   256,   128,    32,   2,   2,   16,   16,       8,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        8,             S<32, 8>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   128,   256,    32,   2,   2,   16,   16,       2,       8,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        4,             S<64, 4>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   128,   128,   128,    32,   2,   2,   16,   16,       4,       4,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 4>,                        8,             S<32, 4>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   256,   128,   128,    32,   2,   2,   16,   16,       4,       2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 8>,                        8,             S<32, 8>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v3>,
        DeviceGemmReduce_Wmma_CShuffleV3<     Col,     Row,     Row,  F16,   F16,   F16,      F32,      F32,       F32, ReducePtrsGlobal, PassThrough, PassThrough, PassThrough, ReduceOps, ReduceInElementOps, ReduceOutElementOps,  ReduceMemOp,    GemmDefault,   128,    64,   128,    32,   2,   2,   16,   16,       2,       4,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,                        S<1, 32, 1, 4>,                        8,             S<32, 4>,                         4,                            1, Intrawave, BlockGemmPipelineVersion::v3>
        // clang-format on
        >;

void add_device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_kn_mn_instances(
    std::vector<DeviceGemmReducePtr<0, ReducePtrsGlobal::Size()>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_kn_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
