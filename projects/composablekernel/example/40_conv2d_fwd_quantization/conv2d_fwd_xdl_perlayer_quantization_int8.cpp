// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"

using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using InDataType       = int8_t;
using WeiDataType      = int8_t;
using AccDataType      = int32_t;
using CShuffleDataType = AccDataType;
using OutDataType      = int8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using ActivationOp = PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<ActivationOp>;

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
        64,          // KPerBlock
        16,          // AK1
        16,          // BK1
        16,          // MPerXdl
        16,          // NPerXdl
        4,           // MXdlPerWave
        8,           // NXdlPerWave
        S<4, 64, 1>, // ABlockTransferThreadClusterLengths_AK0_M_AK1
        S<1, 0, 2>,  // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // ABlockTransferSrcAccessOrder
        2,           // ABlockTransferSrcVectorDim
        16,          // ABlockTransferSrcScalarPerVector
        16,          // ABlockTransferDstScalarPerVector_AK1
        1,           // ABlockLdsExtraM
        S<4, 64, 1>, // BBlockTransferThreadClusterLengths_BK0_N_BK1
        S<1, 0, 2>,  // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,  // BBlockTransferSrcAccessOrder
        2,           // BBlockTransferSrcVectorDim
        16,          // BBlockTransferSrcScalarPerVector
        16,          // BBlockTransferDstScalarPerVector_BK1
        1,           // BBlockLdsExtraN
        1,
        1,
        S<1, 32, 1, 8>,
        4>;

#include "run_conv2d_fwd_perlayer_quantization_example.inc"

int main(int argc, char* argv[])
{
    bool do_verification = true;
    bool time_kernel     = false;

    if(argc == 1)
    {
        // use default
    }
    else if(argc == 3)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    float requant_scale       = 0.5f;
    const auto out_element_op = OutElementOp{requant_scale, ActivationOp{}};
    run_conv2d_fwd_perlayer_quantization_example(out_element_op, do_verification, time_kernel);
}
