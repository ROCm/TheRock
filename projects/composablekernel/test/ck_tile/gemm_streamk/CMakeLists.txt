# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)
set(EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})

# Currently test_ck_tile_streamk_smoke is only built on gfx9
if(GPU_TARGETS MATCHES "gfx90a|gfx942|gfx950")

    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
 
    #TODO: support all arches
    #TODO: current c-shuffle only supports C layout as R
    add_gtest_executable(test_ck_tile_streamk_tile_partitioner test_streamk_tile_partitioner.cpp)
    add_gtest_executable(test_ck_tile_streamk_reduction
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_fp16_reduction.cpp
                        test_gemm_streamk_util.cpp)
    add_gtest_executable(test_ck_tile_streamk_smoke 
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_fp16_persistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_bf16_persistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_fp8_persistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_bf8_persistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_fp16_nonpersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_bf16_nonpersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_fp8_nonpersistent.cpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/smoke_tests/test_gemm_streamk_bf8_nonpersistent.cpp
                         test_gemm_streamk_util.cpp)
    add_gtest_executable(test_ck_tile_streamk_extended
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp16_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf16_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp8_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf8_persistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp16_nonpersistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf16_nonpersistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_fp8_nonpersistent.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/extended_tests/test_gemm_streamk_bf8_nonpersistent.cpp
                         test_gemm_streamk_util.cpp)
    target_compile_options(test_ck_tile_streamk_smoke PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    target_compile_options(test_ck_tile_streamk_extended PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
else()
    message(DEBUG "Skipping test_ck_tile_streamk unit tests for current target")
endif()
