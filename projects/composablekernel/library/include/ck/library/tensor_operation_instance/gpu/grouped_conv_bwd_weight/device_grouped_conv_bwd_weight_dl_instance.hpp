// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

static constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_dl_f32_instances =
    std::tuple<
        // clang-format off
        //############################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1|  M1Per|  N1Per|   KPer|  M1N1Thread|  M1N1Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|         ABlockTransfer|     ABlockTransfer|         ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|         BBlockTransfer|     BBlockTransfer|         BBlockTransfer|   CThreadTransfer| CThreadTransfer|    CThreadTransfer| 
        //############################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   | Thread| Thread| Thread| ClusterM1Xs| ClusterN1Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster| SrcAccessOrder| SrcVectorTensorLengths|    SrcVectorTensor| DstVectorTensorLengths| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster| SrcAccessOrder| SrcVectorTensorLengths|    SrcVectorTensor| DstVectorTensorLengths| SrcDstAccessOrder| SrcDstVectorDim| DstScalarPerVector|
        //############################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |       |       |       |            |            |       _K0_M0_M1_K1|         _K0_M0_M1_K1|   ArrangeOrder|               |           _K0_M0_M1_K1| ContiguousDimOrder|           _K0_M0_M1_K1|       _K0_N0_N1_K1|         _K0_N0_N1_K1|   ArrangeOrder|               |           _K0_N0_N1_K1| ContiguousDimOrder|           _K0_N0_N1_K1|                  |                |                   |
        //############################|           |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |       |       |       |            |            |                   |                     |               |               |                       |                   |                       |                   |                     |               |               |                       |                   |                       |                  |                |                   |
        // generic instance
        DeviceGroupedConvBwdWeight_Dl< NDimSpatial,  ALayout,   BLayout,   ELayout,    F32,     F32,     F32,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,   256,   128,   128,    16,  1,      4,      4,      1,     S<8, 2>,     S<8, 2>,   S<1, 8, 1, 1, 1>,   S<1, 2, 1, 128, 1>, S<0, 2, 3, 1, 4>, S<0, 2, 3, 1, 4>,   S<1, 1, 1, 1, 1>,   S<0, 2, 3, 1, 4>,       S<1, 1, 1, 1, 1>,   S<1, 1, 1, 8, 1>,   S<1, 16, 1, 16, 1>, S<0, 1, 4, 2, 3>, S<0, 1, 4, 2, 3>,   S<1, 1, 1, 1, 1>,   S<0, 1, 4, 2, 3>,       S<1, 1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,             5,                   1>
        // clang-format on
        >;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_dl_f16_instances =
    std::tuple<
        // clang-format off
        //############################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1|  M1Per|  N1Per|   KPer|  M1N1Thread|  M1N1Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|         ABlockTransfer|     ABlockTransfer|         ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|         BBlockTransfer|     BBlockTransfer|         BBlockTransfer|   CThreadTransfer| CThreadTransfer|    CThreadTransfer| 
        //############################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   | Thread| Thread| Thread| ClusterM1Xs| ClusterN1Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster| SrcAccessOrder| SrcVectorTensorLengths|    SrcVectorTensor| DstVectorTensorLengths| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster| SrcAccessOrder| SrcVectorTensorLengths|    SrcVectorTensor| DstVectorTensorLengths| SrcDstAccessOrder| SrcDstVectorDim| DstScalarPerVector|
        //############################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |       |       |       |            |            |       _K0_M0_M1_K1|         _K0_M0_M1_K1|   ArrangeOrder|               |           _K0_M0_M1_K1| ContiguousDimOrder|           _K0_M0_M1_K1|       _K0_N0_N1_K1|         _K0_N0_N1_K1|   ArrangeOrder|               |           _K0_N0_N1_K1| ContiguousDimOrder|           _K0_N0_N1_K1|                  |                |                   |
        //############################|           |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |       |       |       |            |            |                   |                     |               |               |                       |                   |                       |                   |                     |               |               |                       |                   |                       |                  |                |                   |
        // generic instance
        DeviceGroupedConvBwdWeight_Dl< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,   256,   128,   128,    16,  1,      4,      4,      1,     S<8, 2>,     S<8, 2>,   S<1, 8, 1, 1, 1>,   S<1, 2, 1, 128, 1>, S<0, 2, 3, 1, 4>, S<0, 2, 3, 1, 4>,   S<1, 1, 1, 1, 1>,   S<0, 2, 3, 1, 4>,       S<1, 1, 1, 1, 1>,   S<1, 1, 1, 8, 1>,   S<1, 16, 1, 16, 1>, S<0, 1, 4, 2, 3>, S<0, 1, 4, 2, 3>,   S<1, 1, 1, 1, 1>,   S<0, 1, 4, 2, 3>,       S<1, 1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,             5,                   1>
        // clang-format on
        >;

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_dl_bf16_instances =
    std::tuple<
        // clang-format off
        //############################|        Num| InLayout| WeiLayout| OutLayout| InData| WeiData| OutData| AccData|          In|         Wei|         Out|              ConvBackward| Block|  MPer|  NPer| K0Per| K1|  M1Per|  N1Per|   KPer|  M1N1Thread|  M1N1Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|         ABlockTransfer|     ABlockTransfer|         ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|         BBlockTransfer|     BBlockTransfer|         BBlockTransfer|   CThreadTransfer| CThreadTransfer|    CThreadTransfer| 
        //############################|        Dim|         |          |          |   Type|    Type|    Type|    Type| Elementwise| Elementwise| Elementwise|                    Weight|  Size| Block| Block| Block|   | Thread| Thread| Thread| ClusterM1Xs| ClusterN1Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster| SrcAccessOrder| SrcVectorTensorLengths|    SrcVectorTensor| DstVectorTensorLengths| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster| SrcAccessOrder| SrcVectorTensorLengths|    SrcVectorTensor| DstVectorTensorLengths| SrcDstAccessOrder| SrcDstVectorDim| DstScalarPerVector|
        //############################|    Spatial|         |          |          |       |        |        |        |   Operation|   Operation|   Operation|            Specialization|      |      |      |      |   |       |       |       |            |            |       _K0_M0_M1_K1|         _K0_M0_M1_K1|   ArrangeOrder|               |           _K0_M0_M1_K1| ContiguousDimOrder|           _K0_M0_M1_K1|       _K0_N0_N1_K1|         _K0_N0_N1_K1|   ArrangeOrder|               |           _K0_N0_N1_K1| ContiguousDimOrder|           _K0_N0_N1_K1|                  |                |                   |
        //############################|           |         |          |          |       |        |        |        |            |            |            |                          |      |      |      |      |   |       |       |       |            |            |                   |                     |               |               |                       |                   |                       |                   |                     |               |               |                       |                   |                       |                  |                |                   |
        // generic instance
        DeviceGroupedConvBwdWeight_Dl< NDimSpatial,  ALayout,   BLayout,   ELayout,   BF16,     F32,    BF16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,   256,   128,   128,    16,  1,      4,      4,      1,     S<8, 2>,     S<8, 2>,   S<1, 8, 1, 1, 1>,   S<1, 2, 1, 128, 1>, S<0, 2, 3, 1, 4>, S<0, 2, 3, 1, 4>,   S<1, 1, 1, 1, 1>,   S<0, 2, 3, 1, 4>,       S<1, 1, 1, 1, 1>,   S<1, 1, 1, 8, 1>,   S<1, 16, 1, 16, 1>, S<0, 1, 4, 2, 3>, S<0, 1, 4, 2, 3>,   S<1, 1, 1, 1, 1>,   S<0, 1, 4, 2, 3>,       S<1, 1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,             5,                   1>
        // clang-format on
        >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
