// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_dpp.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto MNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// Compilation parameters for a[k, m] * b[k, n] = c[m, n]
// clang-format off
using device_gemm_dpp_f16_f16_f16_km_kn_mn_irregular_instances = std::tuple<
    // ########| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer|    MDpp|    NDpp|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CThreadTransfer| CThreadTransfer|
    // ########|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|    |    |  Dpp|  Dpp| PerWave| PerWave|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| SrcDstVectorDim|       DstScalar|
    // ########|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |                |       PerVector|
    // ########|      |      |      |        |        |        |        |            |            |            |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |                |                |
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,   256,   128,   128,    64,   4,   4,   16,   16,       2,       4,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              4,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,   256,    64,    64,    64,   4,   4,   16,   16,       1,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,   128,    64,    64,    64,   4,   4,   32,    8,       2,       2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              4,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,   128,    32,    32,    64,   4,   4,   32,    8,       1,       1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,   128,     4,    32,    64,   4,   4,    1,   32,       1,       1,      S<4, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,    64,    64,    64,    32,   4,   4,   32,    8,       2,       4,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              4,      true,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,    64,    16,    32,    64,   4,   4,    8,   32,       1,       1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,    64,    16,    16,    64,   4,   4,    8,   16,       1,       1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,    64,     2,    32,    64,   4,   4,    1,   32,       1,       1,      S<4, 2, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,    32,    32,    16,    64,   4,   4,   16,   16,       2,       1,      S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              4,      true,      S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,    32,     8,    16,    64,   4,   4,    4,   16,       2,       1,      S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,      S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,               5,               1>,
    DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,      MNPadding,    32,     1,    32,    64,   4,   4,    1,   32,       1,       1,      S<4, 1, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              4,      true,      S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              4,      true,               5,               1>
    >;
// clang-format on

void add_device_gemm_dpp_f16_f16_f16_km_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_gemm_dpp_f16_f16_f16_km_kn_mn_irregular_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
