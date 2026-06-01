// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using BF8  = ck::bf8_t;
using F8   = ck::f8_t;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdDataDefault = ConvolutionBackwardDataSpecialization::Default;

static constexpr auto ConvBwdDataFilter1x1Stride1Pad0 =
    ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0;
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_v3_f16_16_16_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|    DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle |    CShuffle |   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |            |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN|       Size| Block| Block| Block|    |    |  Wmma| Wmma|       |        |     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      MRepeat|    NRepeat  |  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |    |       |       |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|                 |
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |    |       |       |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |                 
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    32,    64,   8,   8,   16,   16,       4,       1,        S<8, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,        S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    64,    32,   8,   8,   16,   16,       4,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              8,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_v3_f16_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|    DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle |    CShuffle |   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |            |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN|       Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      MRepeat|    NRepeat  |  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|                 |
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |                 
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    64,    32,   8,   8,   16,   16,       4,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,        S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<1,1,1>>,
        
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    32,    64,   8,   8,   16,   16,       4,       1,        S<8, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,        S<8, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    64,    32,   8,   8,   16,   16,       4,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              8,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,    64,    32,   8,   8,   16,   16,       8,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,    32,    64,    32,   8,   8,   16,   16,       2,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,   F16,   F16,     F32,      F16, Empty_Tuple,   F16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,    64,    64,   8,   8,   16,   16,       8,       1,       S<8, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_v3_bf16_16_16_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|    DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle |    CShuffle |   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |            |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN|       Size| Block| Block| Block|    |    |  Wmma| Wmma|       |        |     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      MRepeat|    NRepeat  |  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |    |       |       |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|                 |
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |    |       |       |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |                 
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,   BF16, BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,     32,   32,   8,   8,  16,     16,      4,       1,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,   BF16, BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    32,     32,  128,   8,   8,  16,     16,      2,       1,       S<16, 4, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<16, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>
    // clang-format on
    >;

template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          ConvolutionBackwardDataSpecialization ConvSpec>
using device_grouped_conv_bwd_data_wmma_v3_bf16_instances = std::tuple<
    // clang-format off
        // ##############################################|       NDim| ALayout| BLayout|    DsLayout| ELayout| AData| BData| AccData| CShuffle|      DsData| EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MRepeat| NRepeat|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle |    CShuffle |   CDEBlockTransfer| CDEBlockTransfer|
        // ##############################################|    Spatial|        |        |            |        |  Type|  Type|    Type| DataType|        Type|  Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN|       Size| Block| Block| Block|    |    | Wmma| Wmma|        |        |     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      MRepeat|    NRepeat  |  _MBlock_MPerBlock|  ScalarPerVector|
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|                 |
        // ##############################################|           |        |        |            |        |      |      |        |         |            |      |             |             |               |                    |      |      |           |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |                 
        // generic instance
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    64,    32,   8,   8,   16,   16,       4,       2,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,        S<4, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<1,1,1>>,
        
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,   128,    32,   8,   8,   16,   16,       8,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    64,    32,    32,   8,   8,   16,   16,       4,       1,       S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,    32,    64,    32,   8,   8,   16,   16,       2,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,         64,    32,    32,   128,   8,   8,   16,   16,       2,       1,       S<16, 4, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<16, 4, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              8,         1,            1,            1,     S<1, 16, 1, 4>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,   128,    32,   8,   8,   16,   16,       8,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,   128,    32,   8,   8,   16,   16,       8,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,         
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,    64,    32,   8,   8,   16,   16,       8,       1,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,   128,    32,   8,   8,   16,   16,       8,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        128,   128,   128,    32,   8,   8,   16,   16,       8,       2,       S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1,            1,            1,     S<1, 16, 1, 8>,         S<8,8,8>>,
        DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<NDimSpatial,  ALayout, BLayout,   DsLayout, ELayout,  BF16,  BF16,     F32,     BF16, Empty_Tuple,  BF16,  PassThrough,  PassThrough,    PassThrough,            ConvSpec,  true,  true,        256,   128,   128,    32,   8,   8,   16,   16,       8,       1,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,            1,            1,     S<1, 16, 1, 16>,        S<8,8,8>>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
