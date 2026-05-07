# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(GPU_TARGETS MATCHES "gfx9")
    add_custom_target(client_gemm_fastgelu_examples)

    add_executable(client_gemm_add_add_fastgelu gemm_add_add_fastgelu.cpp)
    target_link_libraries(client_gemm_add_add_fastgelu PRIVATE composable_kernel::device_gemm_operations)

    add_executable(client_gemm_add_fastgelu gemm_add_fastgelu.cpp)
    target_link_libraries(client_gemm_add_fastgelu PRIVATE composable_kernel::device_gemm_operations)

    add_executable(client_gemm_fastgelu gemm_fastgelu.cpp)
    target_link_libraries(client_gemm_fastgelu PRIVATE composable_kernel::device_gemm_operations)

    add_dependencies(client_gemm_fastgelu_examples client_gemm_add_add_fastgelu client_gemm_add_fastgelu
                 client_gemm_fastgelu)

    add_custom_target(client_gemm_fastgelu_generic_examples)

    add_executable(client_gemm_add_add_fastgelu_generic gemm_add_add_fastgelu_generic.cpp)
    target_link_libraries(client_gemm_add_add_fastgelu_generic composable_kernel::device_gemm_operations)

    add_executable(client_gemm_add_fastgelu_generic gemm_add_fastgelu_generic.cpp)
    target_link_libraries(client_gemm_add_fastgelu_generic PRIVATE composable_kernel::device_gemm_operations)

    add_executable(client_gemm_fastgelu_generic gemm_fastgelu_generic.cpp)
    target_link_libraries(client_gemm_fastgelu_generic PRIVATE composable_kernel::device_gemm_operations)

    add_dependencies(client_gemm_fastgelu_generic_examples client_gemm_add_add_fastgelu_generic 
                 client_gemm_add_fastgelu_generic client_gemm_fastgelu_generic)
endif()
