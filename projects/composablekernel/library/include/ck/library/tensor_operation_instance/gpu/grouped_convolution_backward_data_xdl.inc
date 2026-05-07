// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_gnhwk_gkyxc_gnhwc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  GNHWK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  GNHWC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f16_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_TF32
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  TF32,
                                                                  TF32>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_tf32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  TF32,
                                                                  TF32>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_tf32_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
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
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_bf16_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NHWGK,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkyxc_ngchw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkyxc_ngchw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkyxc_ngchw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKYXC,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_f32_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv2d_bwd_data_xdl_ngkhw_gkcyx_ngchw_bf16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
                                                                  NGKHW,
                                                                  GKCYX,
                                                                  Empty_Tuple,
                                                                  NGCHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

// conv3d backward data
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_gndhwk_gkzyxc_gndhwc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  GNDHWK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  GNDHWC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f16_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  TF32,
                                                                  TF32>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_tf32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  TF32,
                                                                  TF32>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_f32_tf32_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
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
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_bf16_optimized_loads_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
void add_device_grouped_conv3d_bwd_data_xdl_ndhwgk_gkzyxc_ndhwgc_input_f16_comp_bf8f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  BF8,
                                                                  F8>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkzyxc_ngcdhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkzyxc_ngcdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkzyxc_ngcdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKZYXC,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F16,
                                                                  F16,
                                                                  Empty_Tuple,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f32_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_f32_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  F32,
                                                                  F32,
                                                                  Empty_Tuple,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_bf16_16_16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);

void add_device_grouped_conv3d_bwd_data_xdl_ngkdhw_gkczyx_ngcdhw_bf16_vec_transpose_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NGKDHW,
                                                                  GKCZYX,
                                                                  Empty_Tuple,
                                                                  NGCDHW,
                                                                  BF16,
                                                                  BF16,
                                                                  Empty_Tuple,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
