# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_gtest_executable(test_convnd_bwd_data convnd_bwd_data_xdl.cpp)
if(result EQUAL 0)
   target_link_libraries(test_convnd_bwd_data PRIVATE utility device_conv1d_bwd_data_instance device_conv2d_bwd_data_instance device_conv3d_bwd_data_instance)
endif()
