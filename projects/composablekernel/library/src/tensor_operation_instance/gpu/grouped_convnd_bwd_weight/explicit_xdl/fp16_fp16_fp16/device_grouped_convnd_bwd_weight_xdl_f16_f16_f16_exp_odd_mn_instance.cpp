// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_exp_gemm_xdl_universal_km_kn_mn_instance.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_explicit_gemm_device_operation_instances<
        2,
        NHWGC,
        GKYXC,
        NHWGK,
        F16,
        F16,
        F16,
        PassThrough,
        PassThrough,
        PassThrough,
        device_gemm_xdl_universal_km_kn_mn_irregular_odd_mn_instances<F16,
                                                                      Intrawave,
                                                                      GemmMNKPadding>>(instances);
}

void add_device_grouped_convnd_bwd_weight_xdl_f16_f16_f16_exp_odd_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
    add_explicit_gemm_device_operation_instances<
        3,
        NDHWGC,
        GKZYXC,
        NDHWGK,
        F16,
        F16,
        F16,
        PassThrough,
        PassThrough,
        PassThrough,
        device_gemm_xdl_universal_km_kn_mn_irregular_odd_mn_instances<F16,
                                                                      Intrawave,
                                                                      GemmMNKPadding>>(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
