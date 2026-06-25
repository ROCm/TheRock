// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp"

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance = ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Dl<
    NDimSpatial, // NDimSpatial
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                  ck::tensor_layout::convolution::GNHWC,
                                  ck::tensor_layout::convolution::GNDHWC>>, // InLayout
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                  ck::tensor_layout::convolution::GKYXC,
                                  ck::tensor_layout::convolution::GKZYXC>>, // WeiLayout
    ck::tuple_element_t<NDimSpatial - 1,
                        ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                  ck::tensor_layout::convolution::GNHWK,
                                  ck::tensor_layout::convolution::GNDHWK>>, // OutLayout
    InDataType,                                                             // InDataType
    WeiDataType,                                                            // WeiDataType
    OutDataType,                                                            // OutDataType
    AccDataType,                                                            // AccDataType
    InElementOp,          // InElementwiseOperation
    WeiElementOp,         // WeiElementwiseOperation
    OutElementOp,         // OutElementwiseOperation
    ConvBwdWeightDefault, // ConvBackwardWeightSpecialization
    256,                  // BlockSize
    128,                  // MPerBlock
    128,                  // NPerBlock
    16,                   // K0PerBlock
    2,                    // K1
    4,                    // M1PerThread
    4,                    // N1PerThread
    1,                    // KPerThread
    S<8, 2>,              // M1N1ThreadClusterM1Xs
    S<8, 2>,              // M1N1ThreadClusterN1Xs
    S<1, 8, 1, 1, 2>,     // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
    S<1, 2, 1, 128, 1>,   // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
    S<0, 2, 3, 1, 4>,     // ABlockTransferThreadClusterArrangeOrder
    S<0, 2, 3, 1, 4>,     // ABlockTransferSrcAccessOrder
    S<1, 1, 1, 1, 1>,     // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
    S<0, 2, 3, 1, 4>,     // ABlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 1, 1>,     // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
    S<1, 1, 1, 8, 2>,     // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
    S<1, 16, 1, 16, 1>,   // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
    S<0, 1, 4, 2, 3>,     // BBlockTransferThreadClusterArrangeOrder
    S<0, 1, 4, 2, 3>,     // BBlockTransferSrcAccessOrder
    S<1, 1, 1, 8, 1>,     // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
    S<0, 1, 4, 2, 3>,     // BBlockTransferSrcVectorTensorContiguousDimOrder
    S<1, 1, 1, 1, 2>,     // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
    S<0, 1, 2, 3, 4, 5>,  // CThreadTransferSrcDstAccessOrder
    5,                    // CThreadTransferSrcDstVectorDim
    4>;                   // CThreadTransferDstScalarPerVector

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
