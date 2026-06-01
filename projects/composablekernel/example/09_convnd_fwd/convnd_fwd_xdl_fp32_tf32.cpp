// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#define EXAMPLE_WITH_COMPUTE_DATATYPE

using InDataType       = float;
using WeiDataType      = float;
using AccDataType      = float;
using CShuffleDataType = float;
using OutDataType      = float;
using ComputeDataType  = ck::tf32_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ck::index_t NDimSpatial, typename InLayout, typename WeiLayout, typename OutLayout>
using DeviceGroupedConvNDFwdInstance =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        NDimSpatial,
        InLayout,         // ALayout
        WeiLayout,        // BLayout
        ck::Tuple<>,      // DsLayout
        OutLayout,        // ELayout
        InDataType,       // ADataType
        WeiDataType,      // BDataType
        AccDataType,      // AccDataType
        CShuffleDataType, // CShuffleDataType
        ck::Tuple<>,      // DsDataType
        OutDataType,      // EDataType
        InElementOp,      // AElementwiseOperation
        WeiElementOp,     // BElementwiseOperation
        OutElementOp,     // CDEElementwiseOperation
        ConvSpec,         // ConvForwardSpecialization
        GemmSpec,         // GemmSpecialization
        1,                // NumGemmKPrefetchStage
        256,              // BlockSize
        128,              // MPerBlock
        192,              // NPerBlock
        16,               // KPerBlock
        4,                // AK1
        4,                // BK1
        32,               // MPerXdl
        32,               // NPerXdl
        2,                // MXdlPerWave
        3,                // NXdlPerWave
        S<4, 64, 1>,      // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
        2,                // ABlockTransferSrcVectorDim
        4,                // ABlockTransferSrcScalarPerVector
        4,                // ABlockTransferDstScalarPerVector_AK1
        1,                // ABlockLdsExtraM
        S<4, 64, 1>,      // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
        2,                // BBlockTransferSrcVectorDim
        4,                // BBlockTransferSrcScalarPerVector
        4,                // BBlockTransferDstScalarPerVector_BK1
        1,                // BBlockLdsExtraN
        1,                // CShuffleMXdlPerWavePerShuffle
        1,                // CShuffleNXdlPerWavePerShuffle
        S<1, 16, 1, 16>,  // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        4,                // CDEBlockTransferScalarPerVector_NPerBlock
        ComputeDataType,  // AComputeDataType
        ComputeDataType,  // BComputeDataType
        ck::LoopScheduler::Default, // LoopScheduler
        1                           // NumGroupsToMerge
        >;

#include "run_convnd_fwd_example.inc"

int main(int argc, char* argv[]) { return run_convnd_fwd_example(argc, argv) ? 0 : 1; }

#undef EXAMPLE_WITH_COMPUTE_DATATYPE
