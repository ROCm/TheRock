# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if((DTYPES MATCHES "fp32" OR DTYPES MATCHES "fp64") OR NOT DEFINED DTYPES)
    add_gtest_executable(test_contraction test_contraction_xdl.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_contraction PRIVATE utility device_contraction_bilinear_instance device_contraction_scale_instance)
    endif()
    add_gtest_executable(test_contraction_interface test_contraction_interface_xdl.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_contraction_interface PRIVATE utility device_contraction_bilinear_instance device_contraction_scale_instance)
    endif()
endif()
