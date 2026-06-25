# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_grouped_gemm)

# NOTE: We test for XDL/WMMA support here instead of relying on the usual pattern matching in the parent CMakeLists. This is necessary
# as these tests are universal and dont have "xdl" or "wmma" in their name to signify their target arch. But they will fail to link
# the instance library if there's no instances present for the current arch.
if (CK_USE_XDL OR CK_USE_WMMA)
    add_gtest_executable(test_grouped_gemm_splitk test_grouped_gemm_splitk.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_grouped_gemm_splitk PRIVATE utility device_grouped_gemm_instance)
        add_dependencies(test_grouped_gemm test_grouped_gemm_splitk)
    endif()

    add_gtest_executable(test_grouped_gemm_fastgelu test_grouped_gemm_fastgelu.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_grouped_gemm_fastgelu PRIVATE utility device_grouped_gemm_fastgelu_instance)
        add_dependencies(test_grouped_gemm test_grouped_gemm_fastgelu)
    endif()
endif()

add_gtest_executable(test_grouped_gemm_interface test_grouped_gemm_interface_xdl.cpp)
if(result EQUAL 0)
    target_link_libraries(test_grouped_gemm_interface PRIVATE utility device_grouped_gemm_instance)
    add_dependencies(test_grouped_gemm test_grouped_gemm_interface)
endif()
