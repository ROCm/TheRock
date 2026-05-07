// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp"
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

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// Compilation parameters for a[k, m] * b[n, k] = c[m, n]
using device_gemm_wmma_bf16_bf16_bf16_km_nk_mn_instances = std::tuple<
    // clang-format off
        //######################| ALayout| BLayout| CLayout| AData| BData| CData| AccData| CShuffle|           A|           B|           C|          GEMM| NumPrefetch| Block|  MPer|  NPer|  KPer| K1| MPer| NPer|      M|       N|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffle| CShuffle| CShuffleBlockTransfer| CShuffleBlockTransfer|
        //######################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise|Specialization|            |  Size| Block| Block| Block|   | WMMA| WMMA| Repeat|  Repeat|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|  MRepeat|  MRepeat|        ClusterLengths|       ScalarPerVector|
        //######################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|              |            |      |      |      |      |   |     |     |       |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          | PerStore| PerStore|      MBlock_MPerBlock|                      |
        //######################|        |        |        |      |      |      |        |         |            |            |            |              |            |      |      |      |      |   |     |     |       |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |         |         |      NBlock_NPerBlock|                      |
        /* Prefetch 2, consume enormous vgpr resource*/
        // 8 Waves
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           2,   256,   128,   128,    32,  8,   16,   16,      4,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 32, 1,  8>,                      8>,
        // 4 Waves
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           2,   128,   128,    64,    64,  8,   16,   16,      4,       2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 32, 1,  4>,                      8>,
        // 2 Waves
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           2,    64,    64,    32,    32,  8,   16,   16,      4,       1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 16, 1,  4>,                      8>,
        // 1 Wave
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           2,    32,    16,    16,    32,  8,   16,   16,      1,       1,     S<2, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 16, 1,  2>,                      8>,
        /* Prefetch 1, prefer larger KPerBlock value for better latency hiding*/
        // 8 Waves
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,   256,   128,   256,    64,  8,   16,   16,      4,       4,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 32, 1,  8>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,   256,   128,   128,    64,  8,   16,   16,      4,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 32, 1,  8>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,   256,   128,   160,    64,  8,   16,   16,      2,       5,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 64, 1,  4>,                      8>,
        // 4 Waves
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,   128,   128,   128,    32,  8,   16,   16,      4,       4,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 32, 1,  4>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,   128,   256,    64,    64,  8,   16,   16,      8,       2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 32, 1,  4>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,   128,    64,   256,    64,  8,   16,   16,      2,       8,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 32, 1,  4>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,   128,    64,    80,    64,  8,   16,   16,      1,       5,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 64, 1,  2>,                      8>,
        // 2 Waves
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,    64,    16,    64,    64,  8,   16,   16,      1,       2,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 16, 1,  4>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,    64,    64,    32,    64,  8,   16,   16,      4,       1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 16, 1,  4>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,    64,    32,    64,    64,  8,   16,   16,      2,       2,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 16, 1,  4>,                      8>,
        // 1 Wave
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,    32,    16,    32,    64,  8,   16,   16,      1,       2,     S<2, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 16, 1,  2>,                      8>,
        DeviceGemmWmma_CShuffle<      Col,     Col,     Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, PassThrough, GemmMNKPadding,           1,    32,    16,    16,    64,  8,   16,   16,      1,       1,     S<2, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,      true,        1,        1,       S<1, 16, 1,  2>,                      8>
    // clang-format on
    >;

void add_device_gemm_wmma_bf16_bf16_bf16_km_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Col, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(instances, device_gemm_wmma_bf16_bf16_bf16_km_nk_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
