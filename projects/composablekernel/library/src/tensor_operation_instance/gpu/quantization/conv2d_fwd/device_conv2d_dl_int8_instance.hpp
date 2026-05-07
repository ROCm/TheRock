// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "conv2d_quantization_common.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// clang-format off
template <typename InLayout,
          typename WeiLayout,
          typename DsLayout,
          typename OutLayout,
          typename DsDatatype,
          typename OutElementOp,
          ConvolutionForwardSpecialization ConvSpec,
          index_t DstScalarPerVector>
using device_grouped_conv2d_dl_int8_instances =
    std::tuple<
        // ###########################################|        NDim| InData| WeiData|   MultpleD|     OutData|     AccData| InLayout| WeiLayout|   MultipleD| OutLayout|           In|           Wei|           Out|    Convolution|              GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        // ###########################################|     Spatial|   Type|    Type|       Type|        Type|        Type|         |          |      Layout|          |  Elementwise|   Elementwise|   Elementwise|        Forward|    Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        // ###########################################|            |       |        |           |            |            |         |          |            |          |    Operation|     Operation|     Operation| Specialization|                  |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
        // ###########################################|            |       |        |           |            |            |         |          |            |          |             |              |              |               |                  |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK< NDimSpatial, int8_t,  int8_t, DsDatatype,      int8_t,     int32_t, InLayout, WeiLayout,    DsLayout, OutLayout,  PassThrough,   PassThrough,  OutElementOp,       ConvSpec,          GemmSpec,   256,   128,   128,    16,  4,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>, S<0, 1, 2, 3, 4, 5>,               5, DstScalarPerVector>
    >;
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
