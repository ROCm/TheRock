// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp"

using InDataType = BF16;
// bf16 kernel use fp32 atomic add to accumulate Weight tensor into global memory
using WeiDataType = F32;
using OutDataType = BF16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    // clang-format on
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffleV3<
        NDimSpatial,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                      ck::tensor_layout::convolution::GNHWC,
                                      ck::tensor_layout::convolution::GNDHWC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                      ck::tensor_layout::convolution::GKYXC,
                                      ck::tensor_layout::convolution::GKZYXC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                      ck::tensor_layout::convolution::GNHWK,
                                      ck::tensor_layout::convolution::GNDHWK>>,
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
        256,                  // BlockSize
        128,                  // MPerBlock
        128,                  // NPerBlock
        32,                   // K0PerBlock
        8,                    // K1
        32,                   // MPerXdl
        32,                   // NPerXdl
        2,                    // MXdlPerWave
        2,                    // NXdlPerWave
        S<4, 16, 1>,          // ABlockTransferThreadClusterLengths_K0_M_K1
        S<2, 0, 1>,           // ABlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,           // ABlockTransferSrcAccessOrder
        2,                    // ABlockTransferSrcVectorDim
        1,                    // ABlockTransferSrcScalarPerVector
        2,                    // ABlockTransferDstScalarPerVector_K1
        true,                 // ABlockLdsAddExtraM
        S<4, 16, 1>,          // BBlockTransferThreadClusterLengths_K0_N_K1
        S<2, 0, 1>,           // BBlockTransferThreadClusterArrangeOrder
        S<1, 0, 2>,           // BBlockTransferSrcAccessOrder
        2,                    // BBlockTransferSrcVectorDim
        1,                    // BBlockTransferSrcScalarPerVector
        2,                    // BBlockTransferDstScalarPerVector_K1
        true,                 // BBlockLdsAddExtraN
        1,                    // CShuffleMXdlPerWavePerShuffle
        1,                    // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 4>,       // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        128 / (sizeof(WeiDataType) * CHAR_BIT)>; // CBlockTransferScalarPerVector_NWaveNPerXdl
                                                 // clang-format off

template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    switch(conv_param.num_dim_spatial_)
    {
    case 1: return !run_grouped_conv_bwd_weight<1>(config, conv_param);
    case 2: return !run_grouped_conv_bwd_weight<2>(config, conv_param);
    case 3: return !run_grouped_conv_bwd_weight<3>(config, conv_param);
    default: break;
    }

    return 1;
}
