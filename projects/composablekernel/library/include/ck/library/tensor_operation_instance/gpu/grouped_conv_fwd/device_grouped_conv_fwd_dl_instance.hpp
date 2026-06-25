// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

static constexpr auto ConvFwdOddC =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <typename InLayout,
          typename WeiLayout,
          typename DsLayout,
          typename OutLayout,
          typename DsDatatype,
          typename CDEElementOp,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv2d_fwd_dl_f16_instances = std::tuple<
    // clang-format off
           // ########################################|        NDim| InData| WeiData|    MultpleD| OutData| AccData| InLayout| WeiLayout| MultipleD| OutLayout|          In|          Wei|           Out|    Convolution|              GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
           // ########################################|     Spatial|   Type|    Type|        Type|    Type|    Type|         |          |    Layout|          | Elementwise|  Elementwise|   Elementwise|        Forward|    Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
           // ########################################|            |       |        |            |        |        |         |          |          |          |   Operation|    Operation|     Operation| Specialization|                  |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
           // ########################################|            |       |        |            |        |        |         |          |          |          |            |             |              |               |                  |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        // generic instances
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<           2,    F16,     F16,  DsDatatype,     F16,     F32, InLayout, WeiLayout,  DsLayout, OutLayout, PassThrough,  PassThrough,  CDEElementOp,       ConvSpec,    GemmMNKPadding,     8,    16,     4,     2,  1,          1,          2,      1,       S<4, 2>,       S<1, 1>,      S<2, 1, 2, 1>,        S<1, 1, 8, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 1, 1, 1>,        S<2, 1, 4, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,               5,                 1>,
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<           2,    F16,     F16,  DsDatatype,     F16,     F32, InLayout, WeiLayout,  DsLayout, OutLayout, PassThrough,  PassThrough,  CDEElementOp,       ConvSpec,    GemmMNKPadding,   256,   128,   128,    16,  1,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 1>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<8, 1, 1, 1>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,               5,                 1>,

        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<           2,    F16,     F16,  DsDatatype,     F16,     F32, InLayout, WeiLayout,  DsLayout, OutLayout, PassThrough,  PassThrough,  CDEElementOp,       ConvSpec,    GemmMNKPadding,   256,   128,   128,    16,  2,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>,      S<8, 1, 1, 2>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 2>,      S<1, 2, 0, 3>,       S<1, 1, 1, 2>, S<0, 1, 2, 3, 4, 5>,               5,                 4>
    // clang-format on
    >;

template <typename InLayout,
          typename WeiLayout,
          typename DsLayout,
          typename OutLayout,
          typename DsDatatype,
          typename CDEElementOp,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv2d_fwd_dl_f32_instances = std::tuple<
    // clang-format off
        // clang-format off
           // ########################################|        NDim| InData| WeiData|    MultpleD| OutData| AccData| InLayout| WeiLayout| MultipleD| OutLayout|          In|          Wei|           Out|    Convolution|              GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
           // ########################################|     Spatial|   Type|    Type|        Type|    Type|    Type|         |          |    Layout|          | Elementwise|  Elementwise|   Elementwise|        Forward|    Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
           // ########################################|            |       |        |            |        |        |         |          |          |          |   Operation|    Operation|     Operation| Specialization|                  |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
           // ########################################|            |       |        |            |        |        |         |          |          |          |            |             |              |               |                  |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        // generic instances
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<           2,    F32,     F32,  DsDatatype,     F32,     F32, InLayout, WeiLayout,  DsLayout, OutLayout, PassThrough, PassThrough,  CDEElementOp,       ConvSpec,    GemmMNKPadding,     8,    16,      4,    2,  1,          1,          2,      1,       S<4, 2>,       S<1, 1>,      S<2, 1, 2, 1>,      S<1, 1, 8, 1>,    S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 1, 1, 1>,        S<2, 1, 4, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,               5,                 1>,
        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<           2,    F32,     F32,  DsDatatype,     F32,     F32, InLayout, WeiLayout,  DsLayout, OutLayout, PassThrough, PassThrough,  CDEElementOp,       ConvSpec,    GemmMNKPadding,   256,   128,   128,    16,  1,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 1>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<8, 1, 1, 1>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,               5,                 1>,

        DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<           2,    F32,     F32,  DsDatatype,     F32,     F32, InLayout, WeiLayout,  DsLayout, OutLayout, PassThrough, PassThrough,  CDEElementOp,       ConvSpec,    GemmMNKPadding,   256,   128,   128,    16,  1,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 1>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<8, 1, 1, 1>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>, S<0, 1, 2, 3, 4, 5>,               5,                 4>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
