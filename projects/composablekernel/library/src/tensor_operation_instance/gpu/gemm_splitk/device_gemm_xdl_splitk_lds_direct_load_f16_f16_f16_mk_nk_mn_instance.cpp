// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle_lds_direct_load.hpp"

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

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding  = ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
using device_gemm_xdl_splitk_lds_direct_load_f16_f16_f16_mk_nk_mn_instances = std::tuple<
    // clang-format off
        //#######################################|AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| NumGemmK| Block|  MPer|  NPer|  KPer|  K1| MPer| NPer| MXdl| NXdl|         ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|         BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //#######################################| Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization| Prefetch|  Size| Block| Block| Block|    |  XDL|  XDL|  Per|  Per|          ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar| AddExtraM|          ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //#######################################|     |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |    Stage|      |      |      |      |    |     |     | Wave| Wave| Lengths_KBatch_K0_M_K1|               |               |      PerVector|          | Lengths_KBatch_K0_N_K1|               |              |      PerVector|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //#######################################|     |      |      |        |        |        |        |            |            |            |              |         |      |      |      |      |    |     |     |     |     |                       |               |               |               |          |                       |               |              |               |          |            |            |                                 |                |
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefault,       1,   256,    16,   128,     8,   8,   16,   16,    1,    2,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                  S<1, 16, 1, 16>,               4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefault,       1,    64,    16,    16,     8,   8,   16,   16,    1,    1,          S<1, 8, 2, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 2, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                   S<1, 16, 1, 4>,               4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefault,       1,    64,    16,    16,     8,   8,   16,   16,    1,    1,          S<1, 8, 2, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 2, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                   S<1, 16, 1, 4>,               4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefault,       2,    64,    16,    16,     16,  8,   16,   16,    1,    1,          S<1, 16, 1, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 1, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                   S<1, 16, 1, 4>,               4>,

        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       1,   256,   128,   128,     8,   8,   32,   32,    2,    2,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                   S<1, 16, 1, 16>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       1,   256,    32,    32,     8,   8,   16,   16,    1,    1,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       1,   256,    16,    64,     16,  8,   16,   16,    1,    1,          S<1, 16, 4, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 4, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       1,   128,    16,    64,     16,  8,   16,   16,    1,    2,          S<1, 16, 2, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 2, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       1,   128,    16,    32,     8,   8,   16,   16,    1,    1,          S<1, 8, 4, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 4, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       1,   128,    16,    32,     4,   8,   16,   16,    1,    1,          S<1, 4, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 4, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                     S<1, 8, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       1,    64,    16,    16,     16,  8,   16,   16,    1,    1,          S<1, 16, 1, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 1, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 4>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       2,   256,    64,    16,     8,   8,   16,   16,    1,    1,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 32, 1, 4>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       2,   256,    16,    64,     8,   8,   16,   16,    1,    1,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,       2,    64,    16,    16,     16,  8,   16,   16,    1,    1,          S<1, 16, 1, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 1, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 4>,              4>,

        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       1,   256,   128,   128,     8,   8,   32,   32,    2,    2,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                   S<1, 16, 1, 16>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       1,   256,    16,   128,     16,  8,   16,   16,    1,    2,          S<1, 16, 4, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 4, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                   S<1, 16, 1, 16>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       1,   256,    32,    32,     16,  8,   16,   16,    1,    1,          S<1, 16, 4, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 4, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       1,   256,    32,    32,     8,   8,   16,   16,    1,    1,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       1,   256,    16,    64,     8,   8,   16,   16,    1,    1,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       1,   128,    16,    32,     8,   8,   16,   16,    1,    1,          S<1, 8, 4, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 4, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 8>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       1,    64,    16,    16,     16,  8,   16,   16,    1,    1,          S<1, 16, 1, 4>, S<0, 2, 1, 3>,              3,              2,         0,         S<1, 16, 1, 4>, S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 16, 1, 4>,              4>,
        DeviceGemmXdlSplitKCShuffle_LdsDirectLoad<  F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough, GemmMNKPadding,       2,   256,    64,    16,     8,   8,   16,   16,    1,    1,          S<1, 8, 8, 4>,  S<0, 2, 1, 3>,              3,              2,         0,         S<1, 8, 8, 4>,  S<0, 2, 1, 3>,             3,              2,         0,           1,           1,                    S<1, 32, 1, 4>,              4>
    // clang-format on
    >;

void add_device_gemm_xdl_splitk_lds_direct_load_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_gemm_xdl_splitk_lds_direct_load_f16_f16_f16_mk_nk_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
