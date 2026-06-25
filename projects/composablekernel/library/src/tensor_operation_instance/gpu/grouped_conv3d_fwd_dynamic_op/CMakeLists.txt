# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
set(GROUPED_CONV3D_FWD_DYNAMIC_OP
   xdl/device_grouped_conv3d_fwd_xdl_dynamic_op_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_dynamic_op_ndhwgc_gkzyxc_ndhwgk_f16_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_dynamic_op_ndhwgc_gkzyxc_ndhwgk_f32_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_dynamic_op_ndhwgc_gkzyxc_ndhwgk_int8_instance.cpp)

add_instance_library(device_grouped_conv3d_fwd_dynamic_op_instance ${GROUPED_CONV3D_FWD_DYNAMIC_OP})
