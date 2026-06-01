# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_instance_library(device_image_to_column_instance
   device_image_to_column_gnwc_1d_instance.cpp
   device_image_to_column_gnhwc_2d_instance.cpp
   device_image_to_column_gndhwc_3d_instance.cpp
   device_image_to_column_nwgc_1d_instance.cpp
   device_image_to_column_nhwgc_2d_instance.cpp
   device_image_to_column_ndhwgc_3d_instance.cpp
)
