// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_column_to_image_impl.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;
using namespace ck::conv_tensor_rearrange_op;

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

template <ck::index_t NDimSpatial, typename InLayout>
using device_column_to_image_bf16_instances =
    std::tuple<
        // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |          |       |
        // generic instance
        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,    64,    16,    16,   S<8, 8>,     1>,

        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,    64,    32,    32,   S<8, 8>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,    64,    64,    64,   S<8, 8>,     8>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,   128,    32,    64,  S<8, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,   128,    64,   128,  S<8, 16>,     8>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,   256,    64,    64, S<16, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,   256,   128,   128, S<16, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,       BF16,        BF16,   256,   128,   128, S<16, 16>,     8>
        // clang-format on
        >;

template <ck::index_t NDimSpatial, typename InLayout>
using device_column_to_image_f16_instances =
    std::tuple<
        // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |          |       |
        // generic instance
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,    64,    16,    16,   S<8, 8>,     1>,
        
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,    64,    32,    32,   S<8, 8>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,    64,    64,    64,   S<8, 8>,     8>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,   128,    32,    64,  S<8, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,   128,    64,   128,  S<8, 16>,     8>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,   256,    64,    64, S<16, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,   256,   128,   128, S<16, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F16,         F16,   256,   128,   128, S<16, 16>,     8>
        // clang-format on
        >;

template <ck::index_t NDimSpatial, typename InLayout>
using device_column_to_image_f32_instances =
    std::tuple<
        // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |          |       |
        // generic instance
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F32,         F32,    64,    16,    16,   S<8, 8>,     1>,
        
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F32,         F32,    64,    32,    32,   S<8, 8>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F32,         F32,   128,    32,    64,  S<8, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F32,         F32,   256,    64,    64, S<16, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,        F32,         F32,   256,   128,   128, S<16, 16>,     4>
        // clang-format on
        >;

template <ck::index_t NDimSpatial, typename InLayout>
using device_column_to_image_i8_instances =
    std::tuple<
        // clang-format off
        //#####################|        Num| InLayout| InDataType| OutDataType| Block|  MPer|  KPer|    Thread| Scalar|
        //#####################|        Dim|         |           |            |  Size| Block| Block|   Cluster|    Per|
        //#####################|    Spatial|         |           |            |      |      |      |   Lengths| Vector|
        //#####################|           |         |           |            |      |      |      |          |       |
        // generic instance
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,    64,    16,    16,   S<8, 8>,     1>,

        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,    64,    32,    32,   S<8, 8>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,    64,    64,    64,   S<8, 8>,     8>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,   128,    32,    64,  S<8, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,   128,    64,   128,  S<8, 16>,     8>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,   256,    64,    64, S<16, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,   256,   128,   128, S<16, 16>,     4>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,   256,   128,   128, S<16, 16>,     8>,
        DeviceColumnToImageImpl<NDimSpatial, InLayout,     int8_t,      int8_t,   256,   256,   256, S<16, 16>,     16>
        // clang-format on
        >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
