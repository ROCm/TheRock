// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_BF16

void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part1(
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
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part2(
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
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part3(
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
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part4(
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

// void
// add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
//     std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
//                                                                 NHWGC,
//                                                                 GKYXC,
//                                                                 Tuple<>,
//                                                                 NHWGK,
//                                                                 BF16,
//                                                                 BF16,
//                                                                 Tuple<>,
//                                                                 BF16,
//                                                                 PassThrough,
//                                                                 PassThrough,
//                                                                 Clamp>>>& instances);

void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
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
void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part2(
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
void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part3(
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
void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part4(
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

// void
// add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances(
//     std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
//                                                                 NDHWGC,
//                                                                 GKZYXC,
//                                                                 Tuple<>,
//                                                                 NDHWGK,
//                                                                 BF16,
//                                                                 BF16,
//                                                                 Tuple<>,
//                                                                 BF16,
//                                                                 PassThrough,
//                                                                 PassThrough,
//                                                                 Clamp>>>& instances);

#endif

#ifdef CK_ENABLE_FP16

void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part1(
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
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part2(
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
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part3(
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
void add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part4(
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

void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part1(
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
void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part2(
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
void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part3(
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
void add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_ndhwgc_gkzyxc_ndhwgk_f16_instances_part4(
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

// void
// add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instances(
//     std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
//                                                                 NHWGC,
//                                                                 GKYXC,
//                                                                 Tuple<>,
//                                                                 NHWGK,
//                                                                 F16,
//                                                                 F16,
//                                                                 Tuple<>,
//                                                                 F16,
//                                                                 PassThrough,
//                                                                 PassThrough,
//                                                                 Clamp>>>& instances);

// void
// add_device_grouped_conv3d_fwd_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_instances(
//     std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
//                                                                 NDHWGC,
//                                                                 GKZYXC,
//                                                                 Tuple<>,
//                                                                 NDHWGK,
//                                                                 F16,
//                                                                 F16,
//                                                                 Tuple<>,
//                                                                 F16,
//                                                                 PassThrough,
//                                                                 PassThrough,
//                                                                 Clamp>>>& instances);

#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
