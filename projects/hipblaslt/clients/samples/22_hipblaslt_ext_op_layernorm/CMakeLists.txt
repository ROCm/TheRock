# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_ext_op_layernorm ${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_ext_op_layernorm.cpp)
target_link_libraries(sample_hipblaslt_ext_op_layernorm PRIVATE sample-helper)
