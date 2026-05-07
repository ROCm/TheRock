// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// conv1d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           NWGC,
                                                           GKXC,
                                                           NWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           NWGC,
                                                           GKXC,
                                                           NWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv1d_bwd_weight_dl_gnwc_gkxc_gnwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           GNWC,
                                                           GKXC,
                                                           GNWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv1d_bwd_weight_dl_nwgc_gkxc_nwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<1,
                                                           NWGC,
                                                           GKXC,
                                                           NWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
// conv2d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_weight_dl_gnhwc_gkyxc_gnhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           GNHWC,
                                                           GKYXC,
                                                           GNHWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_weight_dl_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<2,
                                                           NHWGC,
                                                           GKYXC,
                                                           NHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
// conv3d backward weight
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           BF16,
                                                           F32,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F16,
                                                           F16,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_dl_gndhwc_gkzyxc_gndhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           GNDHWC,
                                                           GKZYXC,
                                                           GNDHWK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_weight_dl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeight<3,
                                                           NDHWGC,
                                                           GKZYXC,
                                                           NDHWGK,
                                                           F32,
                                                           F32,
                                                           F32,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
