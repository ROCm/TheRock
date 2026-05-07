# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_executable(sample_hipblaslt_groupedgemm_fixed_mk_ext "${CMAKE_CURRENT_SOURCE_DIR}/sample_hipblaslt_groupedgemm_fixed_mk_ext.cpp")
target_link_libraries(sample_hipblaslt_groupedgemm_fixed_mk_ext PRIVATE sample-helper)
