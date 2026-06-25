// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = BF16;
using BDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DDataType        = BF16;
using EDataType        = BF16;

using ALayout = Row;
using BLayout = Col;
using DLayout = Row;
using ELayout = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = AddRelu;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using DeviceOpInstance =
    ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<ALayout,
                                                                   BLayout,
                                                                   ck::Tuple<DLayout>,
                                                                   ELayout,
                                                                   ADataType,
                                                                   BDataType,
                                                                   AccDataType,
                                                                   CShuffleDataType,
                                                                   ck::Tuple<DDataType>,
                                                                   EDataType,
                                                                   AElementOp,
                                                                   BElementOp,
                                                                   CDEElementOp,
                                                                   GemmSpec,
                                                                   1,
                                                                   256,
                                                                   256,
                                                                   128,
                                                                   32,
                                                                   8,
                                                                   8,
                                                                   16,
                                                                   16,
                                                                   8,
                                                                   4,
                                                                   S<4, 64, 1>,
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   8,
                                                                   8,
                                                                   1,
                                                                   S<4, 64, 1>,
                                                                   S<1, 0, 2>,
                                                                   S<1, 0, 2>,
                                                                   2,
                                                                   8,
                                                                   8,
                                                                   1,
                                                                   1,
                                                                   1,
                                                                   S<1, 32, 1, 8>,
                                                                   4>;

#include "run_gemm_add_relu_example_xdl.inc"

int main(int argc, char* argv[]) { return !run_gemm_add_relu_example(argc, argv); }
