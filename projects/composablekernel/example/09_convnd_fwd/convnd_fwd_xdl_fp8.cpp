// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "convnd_fwd_common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"

#define EXAMPLE_WITH_COMPUTE_DATATYPE

using InDataType       = ck::f8_t;
using WeiDataType      = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::f8_t;
using OutDataType      = ck::f8_t;
using ComputeDataType  = ck::f8_t;

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
        InLayout,
        WeiLayout,
        ck::Tuple<>,
        OutLayout,
        InDataType,
        WeiDataType,
        AccDataType,
        CShuffleDataType,
        ck::Tuple<>,
        OutDataType,
        InElementOp,
        WeiElementOp,
        OutElementOp,
        ConvSpec,    // ConvForwardSpecialization
        GemmSpec,    // GemmSpecialization
        1,           //
        256,         // BlockSize
        128,         // MPerBlock
        256,         // NPerBlock
        32,          // KPerBlock
        8,           // AK1
        8,           // BK1
        16,          // MPerXdl
        16,          // NPerXdl
        4,           // MXdlPerWave
        8,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        8,           // ABlockTransferSrcScalarPerVector
        8,           // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        8,           // BBlockTransferSrcScalarPerVector
        8,           // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 32, 1, 8>,
        4,
        ComputeDataType>;

#include "run_convnd_fwd_example.inc"

int main(int argc, char* argv[])
{
    // temp disable on gfx11
    if(ck::is_gfx11_supported())
    {
        return 0;
    }
    return run_convnd_fwd_example(argc, argv) ? 0 : 1;
}

#undef EXAMPLE_WITH_COMPUTE_DATATYPE
