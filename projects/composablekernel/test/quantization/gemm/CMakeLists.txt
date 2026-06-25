# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_gemm_quantization_targets)

# Only build test if the quantization instance library exists
if(TARGET device_quantization_instance)
    add_gtest_executable(test_gemm_quantization test_gemm_quantization.cpp)
    if(result EQUAL 0)
        target_link_libraries(test_gemm_quantization PRIVATE utility device_quantization_instance)
        add_dependencies(test_gemm_quantization_targets test_gemm_quantization)
    endif()
else()
    message(VERBOSE "Skipping test_gemm_quantization - device_quantization_instance not built for current GPU targets")
endif()

add_dependencies(test_quantization test_gemm_quantization_targets)
