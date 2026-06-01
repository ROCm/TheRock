// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This (ifndef) is a hack to use customized behavior for buffer load rather than using default
// setting Don't use this hack unless absolutely necessary!
// FIXME: make the behavior of buffer load a configurable (template) parameter of each device op
#define CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 1

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/gpu/contraction/device_contraction_instance.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// A[m0, m1, k0, k1] * B[n0, n1, k0, k1] + D[m0, m1, n0, n1] = E[m0, m1, n0, n1]
// k/k/n/n are the fast changing dimension for A/B/D/E
using device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance =
    device_contraction_kk_instance<BF16,
                                   BF16,
                                   F32,
                                   BF16,
                                   Empty_Tuple,
                                   BF16,
                                   F32,
                                   PassThrough,
                                   PassThrough,
                                   Scale,
                                   2>;

void add_device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance(
    std::vector<std::unique_ptr<DeviceContractionMultipleD<2,
                                                           2,
                                                           2,
                                                           BF16,
                                                           BF16,
                                                           Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           Scale,
                                                           F32>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_contraction_scale_m2_n2_k2_xdl_c_shuffle_bf16_bf16_bf16_compute_f32_kkn_instance{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
