// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_xdl_cshuffle.hpp"
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
// outout: h[m, n]
// input: a[k, m], b[k, n], d0[m, n], d1[m, n], gamma[n], beta[n]
template <LoopScheduler GemmLoopScheduler, PipelineVersion GemmPipeline>
using device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances = std::tuple<
    // clang-format off
        //#######################################|      A|      B|            Ds|      H| AData| BData| AccData| CShuffle|        DsData| EMeanVarData| GammaData|  BetaData| HData|           A|           B|         CDE|           H|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|          PostShuffle|     PostShuffle|            Layernorm|       Layernorm|     LoopScheduler|     Pipeline|
        //#######################################| Layout| Layout|        Layout| Layout|  Type|  Type|    Type| DataType|          Type|         Type|      Type|      Type|  Type| Elementwise| Elementwise| Elementwise| Elementwise| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| ThreadClusterLengths| ScalarPerVector| ThreadClusterLengths| ThreadSliceSize|                  |             |
        //#######################################|       |       |              |       |      |      |        |         |              |             |          |          |      |   Operation|   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                 _M_N|   _NWaveNPerXdl|                 _M_N|              _M|                  |             |
        //#######################################|       |       |              |       |      |      |        |         |              |             |          |          |      |            |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                     |                |                     |                |                  |             |
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<32, 8>,               8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   256,   128,   256,    32,   8,   8,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<32, 8>,               8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   128,   128,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<16, 8>,               8,             S<16, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<32, 8>,               8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<32, 4>,               8,             S<32, 4>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<16, 8>,               8,             S<16, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,    64,    64,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<16, 4>,               8,             S<16, 4>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<32, 8>,               8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<32, 8>,               8,             S<32, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   128,   128,    32,    32,   8,   8,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<32, 4>,               8,             S<32, 4>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,   128,    32,   128,    32,   8,   8,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<16, 8>,               8,             S<16, 8>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,    64,    64,    32,    32,   8,   8,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<16, 4>,               8,             S<16, 4>,               1, GemmLoopScheduler, GemmPipeline>,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough,    GemmDefault,        1,    64,    32,    64,    32,   8,   8,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,             S<16, 4>,               8,             S<16, 4>,               1, GemmLoopScheduler, GemmPipeline>
    // clang-format on
    >;

// irregular tile size
using device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_irregular_tile_instances =
    std::tuple<
        // clang-format off
        //#######################################|      A|      B|            Ds|      H| AData| BData| AccData| CShuffle|        DsData| EMeanVarData| GammaData|  BetaData| HData|           A|           B|         CDE|           H|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|          PostShuffle|     PostShuffle|            Layernorm|       Layernorm|          LoopScheduler|              Pipeline|
        //#######################################| Layout| Layout|        Layout| Layout|  Type|  Type|    Type| DataType|          Type|         Type|      Type|      Type|  Type| Elementwise| Elementwise| Elementwise| Elementwise| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| ThreadClusterLengths| ScalarPerVector| ThreadClusterLengths| ThreadSliceSize|                       |                      |
        //#######################################|       |       |              |       |      |      |        |         |              |             |          |          |      |   Operation|   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                 _M_N|   _NWaveNPerXdl|                 _M_N|              _M|                       |                      |
        //#######################################|       |       |              |       |      |      |        |         |              |             |          |          |      |            |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                     |                |                     |                |                       |                      |
        // pipeline v1, 1 wave
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough, GemmMNKPadding,        1,    64,    16,    16,    32,   8,   8,   16,   16,    1,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<16, 4>,              1,             S<16, 4>,               1, LoopScheduler::Default,  PipelineVersion::v1>
#if CK_EXPERIMENTAL_INTER_WAVE_INSTANCES
        // pipeline v1, 2 waves
        ,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough, GemmMNKPadding,        1,    64,    16,    16,    32,   8,   8,   16,   16,    1,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<16, 4>,              1,             S<16, 4>,               1, LoopScheduler::Interwave, PipelineVersion::v1>
#endif
#if CK_EXPERIMENTAL_PIPELINE_V2_INSTANCES
        // pipeline v2, 1 wave
        ,
        DeviceGemmMultipleDLayernorm_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,          F16,       F16,       F16,   F16, PassThrough, PassThrough,  AddReluAdd, PassThrough, GemmMNKPadding,        1,    64,    16,    16,    32,   8,   8,   16,   16,    1,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<16, 4>,              1,             S<16, 4>,               1, LoopScheduler::Default,  PipelineVersion::v2>
#endif
        // clang-format on
        >;

void add_device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDLayernorm<Row,
                                                             Col,
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
        device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances<
            LoopScheduler::Default,
            PipelineVersion::v1>{});
#if CK_EXPERIMENTAL_INTER_WAVE_INSTANCES
    add_device_operation_instances(
        instances,
        device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances<
            LoopScheduler::Interwave,
            PipelineVersion::v1>{});
#endif
#if CK_EXPERIMENTAL_PIPELINE_V2_INSTANCES
    add_device_operation_instances(
        instances,
        device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_instances<
            LoopScheduler::Default,
            PipelineVersion::v2>{});
#endif
    add_device_operation_instances(
        instances,
        device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_mk_nk_mn_mn_mn_irregular_tile_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
