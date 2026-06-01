// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;

// clang-format off
template <index_t NDims,
          typename ElementwiseOp>
using device_permute_scale_f16_instances =
    std::tuple <
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 64,  64,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 128, 32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 32,  128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 64,  32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 32,  64,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 16,  128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 128, 16,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  32,  32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  16,  64,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  64,  16,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,  32,  16,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,  16,  32,  4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,

         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 256, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256,  64, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 128, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  64, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  32, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 256, 32,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   64, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   32, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  128, 32,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,   64, 32,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,   32, 64,  8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,

#if 0
        // Disabled instances to improve compilation time
        // They listed here to show other possible combinations of parameters
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 256, 256, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 256, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 128, 256, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  32, 512, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 512, 64,  16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   64, 256, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  256,  64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  128, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,  128,  64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,   64, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,

         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256,  64, 128, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 128,  64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  64,  64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  32, 128, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256,  32, 256, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  16, 256, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 128,  32, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  32,   64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  16,  128, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  64,   32, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,  32,   32, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,  16,   64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,

         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 128,  64, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 256,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256,  64, 128, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 128,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  64,  64, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  32, 128, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 256,  16, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   64,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   32,  64, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,  128,  16, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,   64,  16, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,   32,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
#endif

         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256, 128,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 256,  32, 128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  64,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  32,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128,  16, 128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 128, 128,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   16,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 64,   64,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,   32,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F16>, ck::Tuple<F16>, ElementwiseOp,  NDims, 32,   16,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>

    >;

template <index_t NDims,
          typename ElementwiseOp>
using device_permute_scale_f32_instances = std::tuple<
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 128,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  32, 128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  64,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  32,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  16, 128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 128,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   16,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   64,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   32,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   16,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,

         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 256,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  64, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 128,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  64, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  32, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 256,  32, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   64,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   32, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,  128,  32, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   64,  32, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   32,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,

#if 0
        // Disabled instances to improve compilation time
        // They listed here to show other possible combinations of parameters
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 256, 256, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 256, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 128, 256, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  32, 512, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 512,  64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,  256,  64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   64, 256, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,  128, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,  128,  64, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   64, 128, 16, 16, ck::Sequence<1, 0>, ck::Sequence<16>, ck::Sequence<16>>,

         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  64, 128, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 128,  64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  64,  64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  32, 128, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  32, 256, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  16, 256, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 128,  32, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   32,  64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   16, 128, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   64,  32, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   32,  32, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   16,  64, 4, 8, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,

         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 128,  64, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 256,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 128,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  64,  64, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  64, 128, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  32, 128, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 256,  16, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   64,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   32,  64, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,  128,  16, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   64,  16, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   32,  32, 8, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
#endif

         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256, 128,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 256,  32, 128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  64,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  32,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128,  16, 128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 128, 128,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   16,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 64,   64,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   32,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F32>, ElementwiseOp,  NDims, 32,   16,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>
    >;

#ifdef CK_ENABLE_FP8
template <index_t NDims,
          typename ElementwiseOp>
using device_permute_scale_f32_f8_instances = std::tuple<
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256, 128,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256,  32, 128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  64,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  32,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  16, 128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128, 128,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   16,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   64,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 32,   32,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 32,   16,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<4>, ck::Sequence<4>>,

         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256, 128, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256, 256,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256,  64, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128, 128,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  64, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  32, 256, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128, 256,  32, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   64,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   32, 128, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,  128,  32, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 32,   64,  32, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 32,   32,  64, 8, 8, ck::Sequence<1, 0>, ck::Sequence<8>, ck::Sequence<8>>,

         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256,  64,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256, 128,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 256,  32, 128, 4, 4,  ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  64,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  32,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128,  16, 128, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 128, 128,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   32,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   16,  64, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 64,   64,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 32,   32,  16, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>,
         DeviceElementwiseImpl<ck::Tuple<F32>, ck::Tuple<F8>, ElementwiseOp,  NDims, 32,   16,  32, 4, 4, ck::Sequence<1, 0>, ck::Sequence<1>, ck::Sequence<1>>
    >;
#endif
// clang-format on

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
