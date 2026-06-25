// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle_v3_large_tensor.hpp"
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
using I8   = int8_t;
using I32  = int32_t;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <index_t NDSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataType   = Empty_Tuple,
          typename CDEElementOp = PassThrough>
using device_grouped_conv_fwd_wmma_large_tensor_f16_instances = std::tuple<
    // clang-format off
        //########################################################|    NumDim|       A|       B|       Ds|       E| AData| BData| AccData| CShuffle|         Ds|  EData|            A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################################|   Spatial|  Layout|  Layout|   Layout|  Layout|  Type|  Type|    Type| DataType|   DataType|   Type|  Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################################|          |        |        |         |        |      |      |        |         |           |       |    Operation|   Operation|    Operation|               |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,   F16,   F16,     F32,      F16, DsDataType,    F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,  8,    16,   16,       4,       2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,   F16,   F16,     F32,      F16, DsDataType,    F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,    64,    64,    64,  8,    16,   16,       2,       1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,   F16,   F16,     F32,      F16, DsDataType,    F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   128,   128,    32,  8,    16,   16,       4,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,   F16,   F16,     F32,      F16, DsDataType,    F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,    64,  8,    16,   16,       2,       2,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,   F16,   F16,     F32,      F16, DsDataType,    F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,    96,    64,  8,    16,   16,       2,       3,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               1>
    // clang-format on
    >;

template <index_t NDSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionForwardSpecialization ConvSpec,
          typename DsDataType   = Empty_Tuple,
          typename CDEElementOp = PassThrough>
using device_grouped_conv_fwd_wmma_large_tensor_bf16_instances = std::tuple<
    // clang-format off
        //########################################################|    NumDim|       A|       B|       Ds|       E| AData| BData| AccData| CShuffle|         Ds|  EData|            A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################################|   Spatial|  Layout|  Layout|   Layout|  Layout|  Type|  Type|    Type| DataType|   DataType|   Type|  Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################################|          |        |        |         |        |      |      |        |         |           |       |    Operation|   Operation|    Operation|               |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, DsDataType,   BF16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,    64,  8,    16,   16,       4,       2,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, DsDataType,   BF16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,    64,    64,    64,  8,    16,   16,       2,       1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, DsDataType,   BF16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   128,   128,    32,  8,    16,   16,       4,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, DsDataType,   BF16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,    64,  8,    16,   16,       2,       2,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle_V3_Large_Tensor<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, DsDataType,   BF16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,    96,    64,  8,    16,   16,       2,       3,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 64, 1, 4>,               1>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
