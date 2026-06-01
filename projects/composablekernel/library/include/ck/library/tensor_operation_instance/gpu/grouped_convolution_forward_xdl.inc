// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16
// grouped conv1d forward, GNWC/GKXC/GNWK
void add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<1,
                                                                GNWC,
                                                                GKXC,
                                                                Empty_Tuple,
                                                                GNWK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<1,
                                                                GNWC,
                                                                GKXC,
                                                                Empty_Tuple,
                                                                GNWK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<1,
                                                                GNWC,
                                                                GKXC,
                                                                Empty_Tuple,
                                                                GNWK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv1d_fwd_xdl_gnwc_gkxc_gnwk_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<1,
                                                                GNWC,
                                                                GKXC,
                                                                Empty_Tuple,
                                                                GNWK,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
// grouped conv2d forward, GNHWC/GKYXC/GNHWK
void add_device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
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

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_fwd_xdl_gnhwc_gkyxc_gnhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                GNHWC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                GNHWK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

// grouped conv2d forward, NHWGC/GKYXC/NHWGK
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
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

void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f16_16x16_instances(
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
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                TF32,
                                                                TF32>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_int8_instances(
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
#endif

// grouped conv2d forward, NGCHW/GKYXC/NGKHW
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkyxc_ngkhw_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                int8_t,
                                                                int8_t,
                                                                Empty_Tuple,
                                                                int8_t,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

// grouped conv2d forward, NGCHW/GKCYX/NGKHW
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKCYX,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKCYX,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKCYX,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKCYX,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKCYX,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv2d_fwd_xdl_ngchw_gkcyx_ngkhw_f32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NGCHW,
                                                                GKCYX,
                                                                Empty_Tuple,
                                                                NGKHW,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
// grouped conv3d forward, GNDHWC/GKZYXC/GNDHWK
void add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f16_instances(
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

#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                GNDHWC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                GNDHWK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv3d_fwd_xdl_gndhwc_gkzyxc_gndhwk_int8_instances(
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
#endif

#ifdef CK_ENABLE_BF16
// grouped conv3d forward, NDHWGC/GKZYXC/NDHWGK
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
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

void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16x16_instances(
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
#endif

#ifdef CK_ENABLE_FP8
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_f8_instances(
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
                                                                PassThrough,
                                                                F8>>>& instances);

void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F8,
                                                                F8,
                                                                Empty_Tuple,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                F8>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                TF32,
                                                                TF32>>>& instances);
#endif

#ifdef CK_ENABLE_INT8
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_int8_instances(
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

#ifdef CK_ENABLE_BF8
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                BF8,
                                                                BF8,
                                                                Empty_Tuple,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                BF8>>>& instances);
#endif

#if(defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8))
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f8_bf8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                F8,
                                                                BF8,
                                                                Empty_Tuple,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                F8,
                                                                BF8>>>& instances);
#endif

#if(defined(CK_ENABLE_FP8) && defined(CK_ENABLE_BF8))
void add_device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_bf8_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Empty_Tuple,
                                                                NDHWGK,
                                                                BF8,
                                                                F8,
                                                                Empty_Tuple,
                                                                F8,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough,
                                                                BF8,
                                                                F8>>>& instances);
#endif

// grouped conv3d forward, NGCDHW/GKCZYX/NGKDHW
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                BF16,
                                                                BF16,
                                                                Empty_Tuple,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);

void add_device_grouped_conv3d_fwd_xdl_ngcdhw_gkczyx_ngkdhw_f32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NGCDHW,
                                                                GKCZYX,
                                                                Empty_Tuple,
                                                                NGKDHW,
                                                                F32,
                                                                F32,
                                                                Empty_Tuple,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
