// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16 = ck::half_t;
using F32 = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using device_transpose_f16_instances = std::tuple<
    // clang-format off
    DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
    DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5, 64,   64,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
    DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
    DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
    DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
    DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, PassThrough, 5, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>
    // clang-format on
    >;

using device_transpose_f32_instances = std::tuple<
    // clang-format off
    DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
    DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5, 64,   64,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
    DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
    DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
    DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
    DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, PassThrough, 5, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
