// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_2x_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_part2_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_bf16_direct_load_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                BF16,
                                                                BF16,
                                                                Tuple<>,
                                                                BF16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

#endif

#ifdef CK_ENABLE_FP16

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_comp_2x_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_comp_part2_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f16_direct_load_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F16,
                                                                F16,
                                                                Tuple<>,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

#endif

#ifdef CK_ENABLE_FP32

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp>>>& instances);

#endif

#ifdef CK_ENABLE_TF32
void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv2d_fwd_clamp_xdl_large_tensor_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv2d_fwd_clamp_xdl_merged_groups_nhwgc_gkyxc_nhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv2d_fwd_clamp_xdl_nhwgc_gkyxc_nhwgk_f32_tf32_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Tuple<>,
                                                                NHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_16x16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_comp_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_intra_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);
void add_device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_inter_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                                NDHWGC,
                                                                GKZYXC,
                                                                Tuple<>,
                                                                NDHWGK,
                                                                F32,
                                                                F32,
                                                                Tuple<>,
                                                                F32,
                                                                PassThrough,
                                                                PassThrough,
                                                                Clamp,
                                                                TF32,
                                                                TF32>>>& instances);

#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
