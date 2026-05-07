# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_grouped_convnd_bwd_data test_grouped_convnd_bwd_data.cpp)
    target_link_libraries(test_grouped_convnd_bwd_data PRIVATE utility device_grouped_conv2d_bwd_data_instance device_grouped_conv3d_bwd_data_instance)

    add_executable(test_grouped_convnd_bwd_data_large_cases test_grouped_convnd_bwd_data_large_cases.cpp)
    target_compile_options(test_grouped_convnd_bwd_data_large_cases PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_libraries(test_grouped_convnd_bwd_data_large_cases PRIVATE gtest_main getopt::getopt utility device_grouped_conv2d_bwd_data_instance device_grouped_conv3d_bwd_data_instance)

    add_executable(test_grouped_convnd_bwd_data_dataset_xdl test_grouped_convnd_bwd_data_dataset_xdl.cpp)
    target_compile_options(test_grouped_convnd_bwd_data_dataset_xdl PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_libraries(test_grouped_convnd_bwd_data_dataset_xdl PRIVATE gtest_main getopt::getopt utility device_grouped_conv2d_bwd_data_instance device_grouped_conv3d_bwd_data_instance)

    add_executable(test_grouped_conv_bwd_data_bilinear test_grouped_conv_bwd_data_bilinear.cpp)
    target_compile_options(test_grouped_conv_bwd_data_bilinear PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_libraries(test_grouped_conv_bwd_data_bilinear PRIVATE gtest_main getopt::getopt utility device_grouped_conv3d_bwd_data_bilinear_instance)

    add_executable(test_grouped_conv_bwd_data_scale test_grouped_conv_bwd_data_scale.cpp)
    target_compile_options(test_grouped_conv_bwd_data_scale PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_libraries(test_grouped_conv_bwd_data_scale PRIVATE gtest_main getopt::getopt utility device_grouped_conv3d_bwd_data_scale_instance)
endif()
add_gtest_executable(test_grouped_convnd_bwd_data_interface_xdl test_grouped_convnd_bwd_data_interface_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_convnd_bwd_data_interface_xdl PRIVATE utility device_grouped_conv2d_bwd_data_instance)
endif()
add_gtest_executable(test_grouped_convnd_bwd_data_interface_wmma test_grouped_convnd_bwd_data_interface_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_convnd_bwd_data_interface_wmma PRIVATE utility device_grouped_conv2d_bwd_data_instance)
endif()