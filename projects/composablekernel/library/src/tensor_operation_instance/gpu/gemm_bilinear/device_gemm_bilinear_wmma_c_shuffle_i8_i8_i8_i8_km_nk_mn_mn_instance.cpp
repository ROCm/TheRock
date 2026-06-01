// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I8       = std::int8_t;
using I32      = std::int32_t;
using I8_Tuple = ck::Tuple<std::int8_t>;

using Row       = ck::tensor_layout::gemm::RowMajor;
using Col       = ck::tensor_layout::gemm::ColumnMajor;
using Row_Tuple = ck::Tuple<Row>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Bilinear    = ck::tensor_operation::element_wise::Bilinear;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// e[m, n] = bilinear(a[m, k] * b[k, n], d[m, n])
using device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instances = std::tuple<
    // clang-format off
        //################################|      A|      B|        Ds|      E| AData| BData|  AccData| CShuffle|   DsData| EData|           A|           B|         CDE|           GEMM| Prefetch| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //################################| Layout| Layout|    Layout| Layout|  Type|  Type|     Type| DataType|     Type|  Type| Elementwise| Elementwise| Elementwise| Specialization|    Stage|  Size| Block| Block| Block|    | WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //################################|       |       |          |       |      |      |         |         |         |      |   Operation|   Operation|   Operation|               |         |      |      |      |      |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //################################|       |       |          |       |      |      |         |         |         |      |            |            |            |               |         |      |      |      |      |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8, PassThrough, PassThrough,    Bilinear,    GemmDefault,        1,   256,   128,   128,    64,  16,   16,   16,       4,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           2,               S<1, 32, 1, 8>,              16>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8, PassThrough, PassThrough,    Bilinear,    GemmDefault,        1,   128,    64,    64,    64,  16,   16,   16,       2,       2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           2,               S<1, 32, 1, 4>,              16>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8, PassThrough, PassThrough,    Bilinear,    GemmDefault,        1,    64,    32,    32,    64,  16,   16,   16,       1,       2,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           2,               S<1, 32, 1, 2>,              16>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8, PassThrough, PassThrough,    Bilinear,    GemmDefault,        1,    32,    16,    16,    64,  16,   16,   16,       1,       1,     S<4,  8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,
        
        // M/N/K padding
        //################################|      A|      B|        Ds|      E| AData| BData|  AccData| CShuffle|   DsData| EData|            A|           B|         CDE|           GEMM| Prefetch| Block|  MPer|  NPer| K0Per|  K1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //################################| Layout| Layout|    Layout| Layout|  Type|  Type|     Type| DataType|     Type|  Type|  Elementwise| Elementwise| Elementwise| Specialization|    Stage|  Size| Block| Block| Block|    | WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //################################|       |       |          |       |      |      |         |         |         |      |    Operation|   Operation|   Operation|               |         |      |      |      |      |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //################################|       |       |          |       |      |      |         |         |         |      |             |            |            |               |         |      |      |      |      |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,   256,   128,   128,    64,  16,   16,   16,       4,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           2,               S<1, 32, 1, 8>,              16>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,   128,    64,    64,    64,  16,   16,   16,       2,       2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           2,               S<1, 32, 1, 4>,              16>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,    64,    32,    32,    64,  16,   16,   16,       1,       2,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           2,               S<1, 32, 1, 2>,              16>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,    32,    16,    16,    64,  16,   16,   16,       1,       1,     S<4,  8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,             16,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,   256,   128,   128,    64,   8,   16,   16,       4,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,           1,           2,               S<1, 32, 1, 8>,               8>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,   128,    64,    64,    64,   8,   16,   16,       2,       2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,           1,           2,               S<1, 32, 1, 4>,               8>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,    64,    32,    32,    64,   8,   16,   16,       1,       2,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,           1,           2,               S<1, 32, 1, 2>,               8>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,    32,    16,    16,    64,   8,   16,   16,       1,       1,     S<4,  8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,   256,   128,   128,    32,   4,   16,   16,       4,       2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,           1,           2,               S<1, 32, 1, 8>,               4>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,   128,    64,    64,    32,   4,   16,   16,       2,       2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,           1,           2,               S<1, 32, 1, 4>,               4>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,    64,    32,    32,    32,   4,   16,   16,       1,       2,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,           1,           2,               S<1, 32, 1, 2>,               4>,
        DeviceGemmMultipleD_Wmma_CShuffle<     Col,    Col, Row_Tuple,    Row,    I8,    I8,      I32,      I32, I8_Tuple,    I8,  PassThrough, PassThrough,    Bilinear, GemmMNKPadding,        1,    32,    16,    16,    32,   4,   16,   16,       1,       1,     S<4,  8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,           1,           1,               S<1, 16, 1, 2>,               4>

    // clang-format on
    >;

void add_device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Col,
                                                    Col,
                                                    Row_Tuple,
                                                    Row,
                                                    I8,
                                                    I8,
                                                    I8_Tuple,
                                                    I8,
                                                    PassThrough,
                                                    PassThrough,
                                                    Bilinear>>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
