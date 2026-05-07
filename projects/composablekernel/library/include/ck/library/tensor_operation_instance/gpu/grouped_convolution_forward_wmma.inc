// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_i8_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_i8_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_i8_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_nhwgc_gkyxc_nhwgk_f16_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_gndhwc_gkzyxc_gndhwk_f16_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_wmma_ndhwgc_gkzyxc_ndhwgk_f16_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_oddc_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
