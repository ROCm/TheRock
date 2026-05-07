// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

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
using BsDataType       = ck::Tuple<B0DataType, B1DataType>;
using AccDataType      = F32;
using CShuffleDataType = BF16;
using D0DataType       = BF16;
// using DsDataType       = ck::Tuple<D0DataType>;
using EDataType = BF16;

using A0Layout = Row;
using AsLayout = ck::Tuple<A0Layout>;
using B0Layout = Row;
using B1Layout = B0Layout;
using BsLayout = ck::Tuple<B0Layout, B1Layout>;
using D0Layout = Row;
// using DsLayout = ck::Tuple<Row>;
using ELayout = Row;

using Multiply    = ck::tensor_operation::element_wise::Multiply;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;
using Add         = ck::tensor_operation::element_wise::Add;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;

using AElementOp = PassThrough;
using BElementOp = Multiply;
// using CDEElementOp = AddFastGelu;

static constexpr auto GemmDefault    = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmMNPadding  = ck::tensor_operation::device::GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <typename DsLayout,
          typename DsDataType,
          typename CDEElementOp,
          ck::tensor_operation::device::GemmSpecialization GemmSpec>
using device_grouped_gemm_xdl_fixed_nk_multi_abd_bf16_i8_bf16_mk_kn_mn_instances = std::tuple<
    // clang-format off
        //######################################|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|     AccData|         CShuffle|     DsData|     EData|           A|           B|          CDE|           GEMM|NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //######################################|         |         |         |        |       Type|       Type|        Type|         DataType|       Type|      Type| Elementwise| Elementwise|  Elementwise| Spacialization|Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //######################################|         |         |         |        |           |           |            |                 |           |          |   Operation|   Operation|    Operation|               |   Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //######################################|         |         |         |        |           |           |            |                 |           |          |            |            |             |               |        |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,    64,    32,   8,   2,   32,   32,    2,    1,    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,    64,   128,    32,   8,   2,   32,   32,    1,    2,    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,   128,    64,    32,   8,   2,   32,   32,    2,    2,    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    64,   128,    32,   8,   2,   32,   32,    2,    2,    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,         0,           1,           1,               S<1, 16, 1, 8>,               8>,
        DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,        1,   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,         1,           1,           1,               S<1, 16, 1, 8>,               8>
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
