// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3.hpp"
#include "ck/utility/sequence.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;
static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;

static constexpr auto V1 = BlockGemmPipelineVersion::v1;
static constexpr auto V3 = BlockGemmPipelineVersion::v3;

using device_gemm_multiply_add_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances =
    std::tuple<
        // clang-format off
        //##################################| ALayout| BLayout|      DsLayout| ELayout| AData| BData|        DsData| EData| AccData| CShuffle|           A|           B|          CDE|      GemmSpec| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|   CShuffle|   CShuffle| CShuffleBlockTransfer| CDEShuffleBlockTransfer|   BlkGemm|     BlkGemm|
        //##################################|        |        |              |        |  Type|  Type|          Type|  Type|    Type| DataType| Elementwise| Elementwise|  Elementwise|              |  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|    MRepeat|    NRepeat|        ClusterLengths|        ScalarPerVectors| PipeSched| PipelineVer|
        //##################################|        |        |              |        |      |      |              |      |        |         |   Operation|   Operation|    Operation|              |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          | PerShuffle| PerShuffle|     _MBlock_MPerBlock|                        |          |            |
        //##################################|        |        |              |        |      |      |              |      |        |         |            |            |             |              |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |           |           |     _NBlock_NPerBlock|                        |          |            |
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   256,   128,   128,    32,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         0,          1,          1,        S<1, 32, 1, 8>,              S<8, 8, 8>, Intrawave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   128,   128,    64,    64,   8,   8,   16,   16,       4,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         0,          1,          1,        S<1, 32, 1, 4>,              S<8, 8, 8>, Intrawave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   256,   128,   256,    64,   8,   8,   16,   16,       4,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<8, 8, 8>, Intrawave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   256,   128,   128,    64,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<8, 8, 8>, Intrawave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   128,    64,    64,    32,   8,   8,   16,   16,       2,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              4,         1,          1,          1,        S<1, 32, 1, 2>,              S<8, 8, 8>, Intrawave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   256,   128,   128,    32,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<8, 8, 8>, Interwave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   256,   128,   128,    64,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<8, 8, 8>, Interwave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   128,   128,   128,    32,   8,   8,   16,   16,       4,       4,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 4>,              S<8, 8, 8>, Interwave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,    64,    32,    64,    64,   8,   8,   16,   16,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         0,          1,          1,        S<1, 16, 1, 4>,              S<8, 8, 8>, Interwave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   128,    64,    64,    32,   8,   8,   16,   16,       2,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              4,         0,          1,          1,        S<1, 32, 1, 2>,              S<8, 8, 8>, Interwave,          V1>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   256,   128,   128,    32,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<8, 8, 8>, Intrawave,          V3>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   256,   128,   128,    64,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<8, 8, 8>, Intrawave,          V3>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   128,   128,   128,    32,   8,   8,   16,   16,       4,       4,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,          1,          1,        S<1, 32, 1, 4>,              S<8, 8, 8>, Intrawave,          V3>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Row, Row_Row_Tuple,     Row,   F16,   F16, F16_F16_Tuple,   F16,     F32,      F16, PassThrough, PassThrough, MultiplyAdd, GemmMNKPadding,   128,    64,    64,    32,   8,   8,   16,   16,       2,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              4,         1,          1,          1,        S<1, 32, 1, 2>,              S<8, 8, 8>, Intrawave,          V3>
        // clang-format on
        >;

void add_device_gemm_multiply_add_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Row,
                                                          Row_Row_Tuple,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          F16_F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_multiply_add_wmma_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
