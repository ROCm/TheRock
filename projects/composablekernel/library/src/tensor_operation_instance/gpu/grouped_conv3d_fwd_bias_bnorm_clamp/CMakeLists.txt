# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
set(GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP)
include(ShardInstantiation)


set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16x16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_16x16_instance.in
  NUM_SHARDS 4
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16x16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_16x16_instance.in
  NUM_SHARDS 4
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16x16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_16x16_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_16x16_instances
  TEMPLATE_FILE xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_16x16_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl
)
   # large tensor
   # NDHWGC, GKZYXC, NDHWGK

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instances
  TEMPLATE_FILE xdl/large_tensor/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/large_tensor
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_instances
  TEMPLATE_FILE xdl/large_tensor/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/large_tensor
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_instances
  TEMPLATE_FILE xdl/large_tensor/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_instance.in
  NUM_SHARDS 2
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/large_tensor
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances
  TEMPLATE_FILE xdl/large_tensor/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_large_tensor_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.in
  NUM_SHARDS 2
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/large_tensor
)

   # merged groups
   # NDHWGC, GKZYXC, NDHWGK

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_bf16_instances
  TEMPLATE_FILE xdl/merged_groups/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_bf16_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/merged_groups
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f16_instances
  TEMPLATE_FILE xdl/merged_groups/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f16_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/merged_groups
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_instances
  TEMPLATE_FILE xdl/merged_groups/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/merged_groups
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instances
  TEMPLATE_FILE xdl/merged_groups/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_merged_groups_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.in
  NUM_SHARDS 3
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/merged_groups
)
   #mem
   # NDHWGC, GKZYXC, NDHWGK

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_intra_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_intra_instance.in
  NUM_SHARDS 20
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_intra_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_intra_instance.in
  NUM_SHARDS 20
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_intra_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_intra_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)
   # NDHWGC, GKZYXC, NDHWGK

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_intra_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_intra_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_inter_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_mem_inter_instance.in
  NUM_SHARDS 20
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_inter_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_mem_inter_instance.in
  NUM_SHARDS 20
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_inter_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_mem_inter_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_inter_instances
  TEMPLATE_FILE xdl/mem/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_mem_inter_instance.in
  NUM_SHARDS 16
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/mem
)

   #comp
   # NDHWGC, GKZYXC, NDHWGK

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_instance.in
  NUM_SHARDS 11
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_instance.in
  NUM_SHARDS 1
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_comp_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_comp_instance.in
  NUM_SHARDS 4
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_comp_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_comp_instance.in
  NUM_SHARDS 4
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_2x_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_2x_instance.in
  NUM_SHARDS 1
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_2x_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_2x_instance.in
  NUM_SHARDS 1
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_part2_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_bf16_comp_part2_instance.in
  NUM_SHARDS 5
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
generate_sharded_instantiations(
  INSTANCES_NAME device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_part2_instances
  TEMPLATE_FILE xdl/comp/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f16_comp_part2_instance.in
  NUM_SHARDS 12
  SRC_LIST GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP
  OUTPUT_DIR ${GENERATED_DIR}/xdl/comp
)

add_instance_library(device_grouped_conv3d_fwd_bias_bnorm_clamp_instance ${GROUPED_conv3d_FWD_BIAS_BNORM_CLAMP})
