# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_ext_op_amax "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_ext_op_amax.cpp")
target_link_libraries(sample_hipblaslt_ext_op_amax PRIVATE sample-helper)
