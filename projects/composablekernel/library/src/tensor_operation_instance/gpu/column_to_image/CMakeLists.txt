# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_instance_library(device_column_to_image_instance
   device_column_to_image_gnwc_1d_instance.cpp
   device_column_to_image_gnhwc_2d_instance.cpp
   device_column_to_image_gndhwc_3d_instance.cpp
   device_column_to_image_nwgc_1d_instance.cpp
   device_column_to_image_nhwgc_2d_instance.cpp
   device_column_to_image_ndhwgc_3d_instance.cpp
)
