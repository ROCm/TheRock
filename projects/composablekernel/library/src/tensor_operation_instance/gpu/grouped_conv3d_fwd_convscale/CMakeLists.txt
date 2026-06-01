# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY XDL_KERNELS
set(GROUPED_CONV3D_FWD_CONVSCALE
   xdl/device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_f8_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_bf8_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_f8_bf8_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_convscale_ndhwgc_gkzyxc_ndhwgk_bf8_f8_instance.cpp
   xdl/device_grouped_conv3d_fwd_xdl_combconvscale_ndhwgc_gkzyxc_ndhwgk_f8_f8_f32_instance.cpp)

add_instance_library(device_grouped_conv3d_fwd_convscale_instance ${GROUPED_CONV3D_FWD_CONVSCALE})
