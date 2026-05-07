// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_wmma_cshuffle_v3.hpp"

#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using I8   = int8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = BF16;
using AsDataType       = ck::Tuple<A0DataType>;
using B0DataType       = I8;
using B1DataType       = BF16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = BF16;
using EDataType        = BF16;

using A0Layout = Row;
using AsLayout = ck::Tuple<A0Layout>;
using B0Layout = Row;
using B1Layout = B0Layout;
using D0Layout = Row;
using ELayout  = Row;

using Multiply            = ck::tensor_operation::element_wise::Multiply;
using MultiplyAddFastGelu = ck::tensor_operation::element_wise::MultiplyAddFastGelu;
using MultiplyFastGelu    = ck::tensor_operation::element_wise::MultiplyFastGelu;
using MultiplyAdd         = ck::tensor_operation::element_wise::MultiplyAdd;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;
using Add         = ck::tensor_operation::element_wise::Add;

using AElementOp = PassThrough;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding  = ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
template <typename BsLayout,
          typename DsLayout,
          typename BsDataType,
          typename DsDataType,
          typename BElementOp,
          typename CDEElementOp,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched>
using device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_kn_mn_comp_instances = std::tuple<
    // clang-format off
       //###################################|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|   CShuffle|   CShuffle|       CBlockTransferClusterLengths|  CBlockTransfer|                       BlkGemmPipeSched|           BlkGemmPipelineVer|
       //###################################|         |         |         |        |       Type|       Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|    MRepeat|    NRepeat| _MBlock_MPerBlock_NBlock_NPerBlock| ScalarPerVector|                                       |                             |
       //###################################|         |         |         |        |           |           |            |                 |           |          |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          | PerShuffle| PerShuffle|                                   |                |                                       |                             |
       //###################################|         |         |         |        |           |           |            |                 |           |          |            |            |             |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |           |           |                                   |                |                                       |                             |
       DeviceGemmMultipleABD_Wmma_CShuffleV3< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   256,    32,   8,   8,   16,   16,       8,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         0,          1,          1,                     S<1, 32, 1, 8>,      S<8, 8, 8>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
       DeviceGemmMultipleABD_Wmma_CShuffleV3< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   128,   128,    64,   8,   8,   16,   16,       4,       2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         0,          1,          1,                     S<1, 32, 1, 8>,      S<8, 8, 8>,  BlockGemmPipelineScheduler::Intrawave, BlockGemmPipelineVersion::v3>,
       DeviceGemmMultipleABD_Wmma_CShuffleV3< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   128,   256,    32,   8,   8,   16,   16,       4,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         0,          1,          1,                     S<1, 32, 1, 8>,      S<8, 8, 8>,                       BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
       DeviceGemmMultipleABD_Wmma_CShuffleV3< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   128,   128,    64,   8,   8,   16,   16,       4,       2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         0,          1,          1,                     S<1, 32, 1, 8>,      S<8, 8, 8>,                       BlkGemmPipeSched, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;

template <typename BsLayout,
          typename DsLayout,
          typename BsDataType,
          typename DsDataType,
          typename BElementOp,
          typename CDEElementOp,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          BlockGemmPipelineScheduler BlkGemmPipeSched>
using device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_kn_mn_mem_instances = std::tuple<
    // clang-format off
        //###################################|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|       CBlockTransferClusterLengths|  CBlockTransfer|  BlkGemmPipeSched|           BlkGemmPipelineVer|
        //###################################|         |         |         |        |       Type|       Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|     MRepeat|     NRepeat| _MBlock_MPerBlock_NBlock_NPerBlock| ScalarPerVector|                  |                             |
        //###################################|         |         |         |        |           |           |            |                 |           |          |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|                                   |                |                  |                             |
        //###################################|         |         |         |        |           |           |            |                 |           |          |            |            |             |               |      |      |      |      |    |    |     |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                   |                |                  |                             |
        DeviceGemmMultipleABD_Wmma_CShuffleV3< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   128,    64,    64,    32,   8,   8,   16,   16,       2,       2,    S< 4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S< 4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,         0,           1,           1,                     S<1, 32, 1, 2>,      S<8, 8, 8>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
        DeviceGemmMultipleABD_Wmma_CShuffleV3< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,    32,    16,    16,   256,   8,   8,   16,   16,       1,       1,    S<32,  1, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<32,  1, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         0,           1,           1,                     S<1, 16, 1, 2>,      S<8, 8, 8>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>,
        DeviceGemmMultipleABD_Wmma_CShuffleV3< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,    64,    16,    32,   256,   8,   8,   16,   16,       1,       1,    S<32,  2, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         0,    S<32,  2, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         0,           1,           1,                     S<1, 16, 1, 4>,      S<8, 8, 8>,  BlkGemmPipeSched, BlockGemmPipelineVersion::v1>
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
