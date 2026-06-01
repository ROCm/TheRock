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

template <index_t... Is>
using S = Sequence<Is...>;

static constexpr auto GemmDefault    = GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

static constexpr auto V3 = BlockGemmPipelineVersion::v3;
static constexpr auto V1 = BlockGemmPipelineVersion::v1;

template <GemmSpecialization GemmSpec>
using device_gemm_multiply_multiply_wmma_i8_i8_f16_mk_nk_mn_instances = std::tuple<
    // clang-format off
        //##################################| ALayout| BLayout|      DsLayout| ELayout| AData| BData|        DsData| EData| AccData| CShuffle|           A|           B|              CDE| GemmSpec| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|   CShuffle|   CShuffle| CShuffleBlockTransfer| CDEShuffleBlockTransfer|    BlkGemm|     BlkGemm| Compute| Compute|
        //##################################|        |        |              |        |  Type|  Type|          Type|  Type|    Type| DataType| Elementwise| Elementwise|      Elementwise|         |  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|    MRepeat|    NRepeat|        ClusterLengths|        ScalarPerVectors|  PipeSched| PipelineVer|   TypeA|   TypeB|
        //##################################|        |        |              |        |      |      |              |      |        |         |   Operation|   Operation|        Operation|         |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          | PerShuffle| PerShuffle|     _MBlock_MPerBlock|                        |           |            |        |        |
        //##################################|        |        |              |        |      |      |              |      |        |         |            |            |                 |         |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |           |           |     _NBlock_NPerBlock|                        |           |            |        |        |
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   128,   128,    64,    64,   8,   8,   16,   16,       4,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,          1,          1,        S<1, 32, 1, 4>,              S<1, 1, 1>,  Intrawave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   256,   128,   256,    64,   8,   8,   16,   16,       4,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<1, 1, 1>,  Intrawave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   128,    64,   256,    64,   8,   8,   16,   16,       2,       8,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 32, 1, 4>,              S<1, 1, 1>,  Intrawave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,    64,    32,    64,    64,   8,   8,   16,   16,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 16, 1, 4>,              S<1, 1, 1>,  Intrawave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   256,   128,   128,    32,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<1, 1, 1>,  Interwave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   256,   128,   256,    64,   8,   8,   16,   16,       4,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<1, 1, 1>,  Interwave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   256,   128,   160,    64,   8,   8,   16,   16,       2,       5,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 64, 1, 4>,              S<1, 1, 1>,  Interwave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,    64,    32,    64,    64,   8,   8,   16,   16,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 16, 1, 4>,              S<1, 1, 1>,  Interwave,          V1,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   256,   128,   128,    32,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<1, 1, 1>,  Intrawave,          V3,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,   256,   128,   128,    64,   8,   8,   16,   16,       4,       2,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,          1,          1,        S<1, 32, 1, 8>,              S<1, 1, 1>,  Intrawave,          V3,      I8,      I8>,
        DeviceGemmMultipleD_Wmma_CShuffleV3<      Row,     Col, Row_Col_Tuple,     Row,    I8,    I8, F16_F16_Tuple,    F16,    I32,       I32, PassThrough, PassThrough, MultiplyMultiply, GemmSpec,    64,    32,    64,    64,   8,   8,   16,   16,       2,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,          1,          1,        S<1, 16, 1, 4>,              S<1, 1, 1>,  Intrawave,          V3,      I8,      I8>
    // clang-format on
    >;

void add_device_gemm_multiply_multiply_wmma_c_shuffle_i8_i8_f16_km_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
                                                          Col,
                                                          Row_Col_Tuple,
                                                          Row,
                                                          I8,
                                                          I8,
                                                          F16_F16_Tuple,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyMultiply>>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_multiply_multiply_wmma_i8_i8_f16_mk_nk_mn_instances<GemmDefault>{});
    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_wmma_i8_i8_f16_mk_nk_mn_instances<GemmMNKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
