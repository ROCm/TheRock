# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    add_gtest_executable(test_grouped_convnd_fwd test_grouped_convnd_fwd.cpp)
    target_link_libraries(test_grouped_convnd_fwd PRIVATE utility device_grouped_conv1d_fwd_instance device_grouped_conv2d_fwd_instance device_grouped_conv3d_fwd_instance)
    add_gtest_executable(test_grouped_convnd_fwd_bilinear test_grouped_convnd_fwd_bilinear.cpp)
    target_link_libraries(test_grouped_convnd_fwd_bilinear PRIVATE utility device_grouped_conv3d_fwd_bilinear_instance)

    add_gtest_executable(test_grouped_convnd_fwd_scaleadd_ab test_grouped_convnd_fwd_scaleadd_ab.cpp)
    target_link_libraries(test_grouped_convnd_fwd_scaleadd_ab PRIVATE utility device_grouped_conv3d_fwd_scaleadd_ab_instance)

    add_executable(test_grouped_convnd_fwd_dataset_xdl test_grouped_convnd_fwd_dataset_xdl.cpp)
    target_compile_options(test_grouped_convnd_fwd_dataset_xdl PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_libraries(test_grouped_convnd_fwd_dataset_xdl PRIVATE gtest_main getopt::getopt utility device_grouped_conv1d_fwd_instance device_grouped_conv2d_fwd_instance device_grouped_conv3d_fwd_instance)

    add_executable(test_grouped_convnd_fwd_large_cases test_grouped_convnd_fwd_large_cases.cpp)
    target_compile_options(test_grouped_convnd_fwd_large_cases PRIVATE -Wno-global-constructors -Wno-undef)
    target_link_libraries(test_grouped_convnd_fwd_large_cases PRIVATE gtest_main getopt::getopt utility device_grouped_conv1d_fwd_instance device_grouped_conv2d_fwd_instance device_grouped_conv3d_fwd_instance)
endif()

add_gtest_executable(test_grouped_convnd_fwd_multi_ab_interface test_grouped_convnd_fwd_multi_ab_interface.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_convnd_fwd_multi_ab_interface PRIVATE utility)
endif()

add_gtest_executable(test_grouped_convnd_fwd_multi_d_interface_compatibility test_grouped_convnd_fwd_multi_d_interface_compatibility_xdl_wmma.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_convnd_fwd_multi_d_interface_compatibility PRIVATE utility device_grouped_conv3d_fwd_instance)
endif()
