// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding  = ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto GemmKPadding   = ck::tensor_operation::device::GemmSpecialization::KPadding;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::PipelineVersion PipVer,
          ck::LoopScheduler LoopSche>
using device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances = std::tuple<
    // clang-format off
        //#########################|AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //#########################| Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //#########################|     |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //#########################|     |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,    32,     8, 16,   16,   16,    1,    1,  S<1, 8,  8, 2>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche, F16, F8>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,    64,     8, 16,   16,   16,    1,    2,  S<1, 8,  8, 2>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche, F16, F8>,
        DeviceGemmXdlSplitKCShuffle<  F16,    F8,   F16,     F32,     Row,      Col,    Row, PassThrough, PassThrough, PassThrough,      GemmSpec,   128,    16,   128,     8, 16,   16,   16,    1,    4,  S<1, 8,  8, 2>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,              3,              8,              8,      true,  S<1, 8, 16, 1>,  S<0, 2, 1, 3>,  S<0, 2, 1, 3>,             3,             16,             16,      true,           1,           1,                   S<1, 16, 1, 8>,               4,  F16, PipVer, LoopSche, F16, F8>
    // clang-format on
    >;

void add_device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_kpb128_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, F16, F8, F16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    // default
    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmDefault,
            ck::PipelineVersion::v2,
            ck::LoopScheduler::Default>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmDefault,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Interwave>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmDefault,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Default>{});

    // MNKPadding
    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmMNKPadding,
            ck::PipelineVersion::v2,
            ck::LoopScheduler::Default>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmMNKPadding,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Interwave>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmMNKPadding,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Default>{});

    // KPadding
    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmKPadding,
            ck::PipelineVersion::v2,
            ck::LoopScheduler::Default>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmKPadding,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Interwave>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmKPadding,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Default>{});

    // MNPadding
    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmMNPadding,
            ck::PipelineVersion::v2,
            ck::LoopScheduler::Default>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmMNPadding,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Interwave>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_f16_f8_f16_mk_nk_mn_irregular_kpb128_instances<
            GemmMNPadding,
            ck::PipelineVersion::v1,
            ck::LoopScheduler::Default>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
