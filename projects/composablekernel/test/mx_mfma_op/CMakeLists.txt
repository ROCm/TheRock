# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_custom_target(test_mx_mfma)

add_gtest_executable(test_mx_mfma_op mx_mfma_op.cpp)
if(result EQUAL 0)
    target_link_libraries(test_mx_mfma_op PRIVATE utility)
endif()
add_dependencies(test_mx_mfma test_mx_mfma_op)


