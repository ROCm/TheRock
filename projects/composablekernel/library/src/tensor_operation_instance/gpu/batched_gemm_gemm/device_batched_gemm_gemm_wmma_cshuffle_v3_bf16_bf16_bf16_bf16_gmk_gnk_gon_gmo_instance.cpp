// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_wmma_cshuffle_v3.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = GemmSpecialization::Default;
static constexpr auto GemmPadded  = GemmSpecialization::MNKOPadding;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

static constexpr auto PipeVerV1 = BlockGemmPipelineVersion::v1;
static constexpr auto PipeVerV3 = BlockGemmPipelineVersion::v3;

// gemm0: Acc[g, m, n] = A[g, m, k] * B0[g, k, n]
// gemm1: C[g, m, o] = Acc[g, m, n] * B1[g, n, o]
// Note that in some cases the "m, o, n" dimensions are referred to as the "gemm1 m, n, k"
// dimensions instead!
template <GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler PipeSched,
          BlockGemmPipelineVersion PipeVer>
using device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instances =
    std::
        tuple<
            // clang-format off
        //################################| ALayout| B0Layout| B1Layout| CLayout| AData| B0Data| B1Data| CData| AccData| CShuffle|           A|          B0|        Acc0|          B1|           C|           GEMM| Block| Gemm01| Gemm0| Gemm0| Gemm1| Gemm1| AK1| BK1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockTransfer| B0BlockLds|  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|   BlkGemm|     BlkGemm|
        //################################|        |         |         |        |  Type|   Type|   Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Elementwise| Elementwise| Specialization|  Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector| PipeSched| PipelineVer|
        //################################|        |         |         |        |      |       |       |      |        |         |   Operation|   Operation|   Operation|   Operation|   Operation|               |      |  Block| Block| Block| Block| Block|    |    |     |     |     |   Per|   Per|   Per| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|          |            |
        //################################|        |         |         |        |      |       |       |      |        |         |            |            |            |            |            |               |      |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|                |               |               |               |               |               |          |                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |            |            |                             |                |          |            |
        //################################|        |         |         |        |      |       |       |      |        |         |            |            |            |            |            |               |      |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|                |               |               |               |               |               |          |                 |                |                |                |                |                |           |                 |                |                |                |                |                |           |            |            |                             |                |          |            |
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,    32,     16,    64,    64,    64,    64,   8,   8,    8,   16,   16,     1,     4,     4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<2, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<2, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,               S<1, 16, 1, 2>,               8, PipeSched,    PipeVer>,
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,    32,     16,   128,    64,    64,    64,   8,   8,    8,   16,   16,     1,     8,     4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<2, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<2, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,               S<1, 16, 1, 2>,               8, PipeSched,    PipeVer>,
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,    64,     32,   128,    64,    64,    64,   8,   8,    8,   16,   16,     1,     8,     4,     S<2, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<4, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<4, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,               S<1, 32, 1, 2>,               8, PipeSched,    PipeVer>,
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,    64,     32,    64,    64,    64,    64,   8,   8,    8,   16,   16,     1,     4,     4,     S<2, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<4, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<4, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,               S<1, 32, 1, 2>,               8, PipeSched,    PipeVer>,
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,   128,     64,   128,    64,    64,    64,   8,   8,    8,   16,   16,     1,     8,     4,     S<2, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<8, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,               S<1, 64, 1, 2>,               8, PipeSched,    PipeVer>,
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,   128,     64,    64,    64,    64,    64,   8,   8,    8,   16,   16,     1,     4,     4,     S<2, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<8, 16, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,               S<1, 64, 1, 2>,               8, PipeSched,    PipeVer>,
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    128,   128,    64,    64,    64,   8,   8,    8,   16,   16,     1,     8,     4,    S<2, 128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,              S<1, 128, 1, 2>,               8, PipeSched,    PipeVer>,
        DeviceBatchedGemmGemm_Wmma_CShuffleV3<  Row,      Col,      Col,     Row,  BF16,   BF16,   BF16,  BF16,     F32,      F32, PassThrough, PassThrough, PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    128,   128,    64,    64,    64,   8,   8,    8,   16,   16,     1,     8,     4,    S<2, 128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,      S<8, 32, 1>,      S<1, 0, 2>,      S<1, 0, 2>,               2,               8,               8,      false,           1,           1,              S<1, 128, 1, 2>,               8, PipeSched,    PipeVer>
            // clang-format on
            >;

void add_device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instance(
    std::vector<std::unique_ptr<DeviceBatchedGemmGemm<Row,
                                                      Col,
                                                      Col,
                                                      Row,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances)
{
    // clang-format off
    add_device_operation_instances(instances, device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instances<GemmDefault, Intrawave, PipeVerV1>{});
    add_device_operation_instances(instances, device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instances<GemmDefault, Interwave, PipeVerV1>{});
    add_device_operation_instances(instances, device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instances<GemmDefault, Intrawave, PipeVerV3>{});
    add_device_operation_instances(instances, device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instances<GemmPadded, Intrawave, PipeVerV1>{});
    add_device_operation_instances(instances, device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instances<GemmPadded, Interwave, PipeVerV1>{});
    add_device_operation_instances(instances, device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instances<GemmPadded, Intrawave, PipeVerV3>{});
    // clang-format on
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
