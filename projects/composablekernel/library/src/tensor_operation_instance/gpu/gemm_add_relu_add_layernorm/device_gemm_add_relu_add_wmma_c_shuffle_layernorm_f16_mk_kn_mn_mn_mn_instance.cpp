// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16           = ck::half_t;
using F32           = float;
using F16_F16_Tuple = ck::Tuple<F16, F16>;

using Row           = ck::tensor_layout::gemm::RowMajor;
using Col           = ck::tensor_layout::gemm::ColumnMajor;
using Row_Row_Tuple = ck::Tuple<Row, Row>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddReluAdd  = ck::tensor_operation::element_wise::AddReluAdd;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// e = elementwise((a * b), d0, d1)
// h = layernorm(e, gamma, beta)
// output: h[m, n]
// input: a[k, m], b[k, n], d0[m, n], d1[m, n], gamma[n], beta[n]
template <BlockGemmPipelineScheduler GemmLoopScheduler, BlockGemmPipelineVersion GemmPipeline>
using device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instances = std::tuple<
    // clang-format off
        //##########################################|      A|      B|            Ds|      H| AData| BData|          DsData| HData| AccData| CShuffleData | EMeanVarData| GammaData|  BetaData|           A|           B|         CDE|           H|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|  CShuffleBlockTransfer|     CDEShuffleBlockTransfer|            Layernorm|       Layernorm|     LoopScheduler|     Pipeline|
        //##########################################| Layout| Layout|        Layout| Layout|  Type|  Type|            Type|  Type|    Type|         Type |         Type|      Type|      Type| Elementwise| Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|         ClusterLengths|            ScalarPerVectors| ThreadClusterLengths| ThreadSliceSize|                  |             |
        //##########################################|       |       |              |       |      |      |                |      |        |              |             |          |          |   Operation|   Operation|   Operation|   Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|  PerVector_BK1|          |  PerShuffle|  PerShuffle|      _MBlock_MPerBlock|                            |                 _M_N|              _M|                  |             |
        //##########################################|       |       |              |       |      |      |                |      |        |              |             |          |          |            |            |            |            |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |      _NBlock_NPerBlock|                            |                     |                |                  |             |
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   256,   128,    32,   2,   2,   16,   16,       8,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   256,   128,    32,   8,   8,   16,   16,       8,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   128,   256,    32,   2,   2,   16,   16,       4,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   128,   256,    32,   8,   8,   16,   16,       4,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   128,   128,   128,    32,   2,   2,   16,   16,       8,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 16, 1, 8>,                           8,             S<16, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   128,   128,   128,    32,   8,   8,   16,   16,       8,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,         S<1, 16, 1, 8>,                           8,             S<16, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   128,   128,    32,   2,   2,   16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   128,   128,    32,   8,   8,   16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   128,   128,    64,    32,   2,   2,   16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 32, 1, 4>,                           8,             S<32, 4>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   128,   128,    64,    32,   8,   8,   16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,         S<1, 32, 1, 4>,                           8,             S<32, 4>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   128,    64,   128,    32,   2,   2,   16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 16, 1, 8>,                           8,             S<16, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   128,    64,   128,    32,   8,   8,   16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,         S<1, 16, 1, 8>,                           8,             S<16, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   128,    64,    32,   2,   2,   16,   16,       4,       1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,    S<16, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,   128,    64,    32,   8,   8,   16,   16,       4,       1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,    64,   128,    32,   2,   2,   16,   16,       2,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,   256,    64,   128,    32,   8,   8,   16,   16,       2,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,         S<1, 32, 1, 8>,                           8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>
    // clang-format on
    >;

// irregular tile size
template <BlockGemmPipelineScheduler GemmLoopScheduler, BlockGemmPipelineVersion GemmPipeline>
using device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_irregular_tile_instances =
    std::tuple<
        // clang-format off
        //##########################################|      A|      B|            Ds|      H| AData| BData|          DsData| HData| AccData| CShuffleData | EMeanVarData| GammaData|  BetaData|           A|           B|         CDE|           H|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|  CShuffleBlockTransfer|     CDEShuffleBlockTransfer|            Layernorm|       Layernorm|     LoopScheduler|     Pipeline|
        //##########################################| Layout| Layout|        Layout| Layout|  Type|  Type|            Type|  Type|    Type|         Type |         Type|      Type|      Type| Elementwise| Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat|         ClusterLengths|            ScalarPerVectors| ThreadClusterLengths| ThreadSliceSize|                  |             |
        //##########################################|       |       |              |       |      |      |                |      |        |              |             |          |          |   Operation|   Operation|   Operation|   Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|  PerVector_BK1|          |  PerShuffle|  PerShuffle|      _MBlock_MPerBlock|                            |                 _M_N|              _M|                  |             |
        //##########################################|       |       |              |       |      |      |                |      |        |              |             |          |          |            |            |            |            |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |      _NBlock_NPerBlock|                            |                     |                |                  |             |
        DeviceGemmMultipleDLayernorm_Wmma_CShuffleV3<    Row,    Row, Row_Row_Tuple,    Row,   F16,   F16,   F16_F16_Tuple,   F16,     F32,           F32,          F16,       F16,       F16, PassThrough, PassThrough,  AddReluAdd, PassThrough, GemmMNKPadding,    64,    32,    32,    32,   8,   8,   16,   16,       2,       1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,         S<1, 16, 1, 4>,                           1,             S<16, 4>,               1, GemmLoopScheduler, GemmPipeline>
        // clang-format on
        >;

void add_device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Row,
                                                             Row,
                                                             Row_Row_Tuple,
                                                             Row,
                                                             F16,
                                                             F16,
                                                             F16_F16_Tuple,
                                                             F16,
                                                             F16,
                                                             F16,
                                                             PassThrough,
                                                             PassThrough,
                                                             AddReluAdd,
                                                             PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_instances<
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v1>{});
    add_device_operation_instances(
        instances,
        device_gemm_add_relu_add_wmma_c_shuffle_layernorm_f16_mk_kn_mn_mn_mn_irregular_tile_instances<
            BlockGemmPipelineScheduler::Intrawave,
            BlockGemmPipelineVersion::v1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
