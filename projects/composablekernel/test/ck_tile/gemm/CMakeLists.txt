# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile_gemm is only built on gfx94/gfx95
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

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx12")
    if(GPU_TARGETS MATCHES "gfx94|gfx95")
        add_gtest_executable(test_ck_tile_gemm_pipeline_mem test_gemm_pipeline_mem.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3 test_gemm_pipeline_compv3.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv4 test_gemm_pipeline_compv4.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_persistent test_gemm_pipeline_persistent.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv6 test_gemm_pipeline_compv6.cpp)

        target_compile_options(test_ck_tile_gemm_pipeline_mem PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv3 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv4 PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_persistent PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv6 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    endif()

    if(GPU_TARGETS MATCHES "gfx95")
        add_gtest_executable(test_ck_tile_gemm_pipeline_comp_async test_gemm_pipeline_comp_async.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_comp_async PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})
    endif()

    if(GPU_TARGETS MATCHES "gfx11|gfx12")
        # On Radeon devices, build the WMMA version instead
        # Define architecture macros for compile-time detection
        if(GPU_TARGETS MATCHES "gfx12")
            list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DARCH_GFX12)
            list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DARCH_GFX12)
        elseif(GPU_TARGETS MATCHES "gfx11")
            list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DARCH_GFX11)
            list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DARCH_GFX11)
        endif()

        add_gtest_executable(test_ck_tile_gemm_pipeline_mem_wmma test_gemm_pipeline_mem_wmma.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3_wmma test_gemm_pipeline_compv3_wmma.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv4_wmma test_gemm_pipeline_compv4_wmma.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_persistent_wmma test_gemm_pipeline_persistent_wmma.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_mem_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv3_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv4_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_persistent_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
    endif()
else()
    message(DEBUG "Skipping ck_tile_gemm tests for current target test_ck_tile_gemm_pipeline") 
endif()
