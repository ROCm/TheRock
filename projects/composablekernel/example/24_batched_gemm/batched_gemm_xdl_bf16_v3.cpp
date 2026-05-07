// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/literals.hpp"

using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

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
using CShuffleDataType = BF16;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    DsDataType,
    EDataType,
    AccDataType,
    CShuffleDataType,
    AElementOp,
    BElementOp,
    CDEElementOp,
    GemmDefault,
    256,            // BlockSize
    256,            // MPerBlock
    128,            // NPerBlock
    32,             // KPerBlock
    8,              // AK1
    8,              // BK1
    16,             // MPerXDL
    16,             // NPerXDL
    8,              // MXdlPerWave
    4,              // NXdlPerWave
    S<4, 64, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
    S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
    2,              // ABlockTransferSrcVectorDim
    8,              // ABlockTransferSrcScalarPerVector
    8,              // ABlockTransferDstScalarPerVector_AK1
    0,              // ABlockLdsExtraM
    S<4, 64, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
    S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
    S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
    2,              // BBlockTransferSrcVectorDim
    8,              // BBlockTransferSrcScalarPerVector
    8,              // BBlockTransferDstScalarPerVector_BK1
    0,              // BBlockLdsExtraN
    1,              // CShuffleMXdlPerWavePerShuffle
    1,              // CShuffleNXdlPerWavePerShuffle
    S<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    S<4>,           // CDEShuffleBlockTransferScalarPerVectors
    ck::BlockGemmPipelineScheduler::Intrawave, // BlockGemmPipelineScheduler
    ck::BlockGemmPipelineVersion::v3           // BlockGemmPipelineVersion
    >;

#include "run_batched_gemm_example.inc"

int main(int argc, char* argv[]) { return run_batched_gemm_example(argc, argv); }
