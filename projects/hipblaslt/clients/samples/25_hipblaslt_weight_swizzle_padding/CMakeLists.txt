# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_weight_swizzle_padding "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_weight_swizzle_padding.cpp")
target_link_libraries(sample_hipblaslt_weight_swizzle_padding PRIVATE sample-helper)
