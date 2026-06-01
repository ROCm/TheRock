# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ============================================================================
# GEMM Tile Engine Unit Tests
# 
# This CMake file creates unit tests for tile_engine generated GEMM kernels.
# It follows the exact same build patterns as tile_engine for consistency
# and reliability. Each kernel configuration gets its own test executable.
# ============================================================================

# Locate tile_engine GEMM scripts directory
set(TILE_ENGINE_GEMM_DIR "${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm")

if(NOT EXISTS ${TILE_ENGINE_GEMM_DIR})
    message(WARNING "Tile engine directory not found: ${TILE_ENGINE_GEMM_DIR}")
    return()
endif()

# ============================================================================
# create_individual_gemm_test_target
#
# Creates a single test executable for a specific kernel configuration.
# Mirrors tile_engine's create_individual_gemm_target function for consistency.
#
# Parameters:
#   datatype     - Data type (fp16, bf16, fp32, etc.)
#   layout       - Matrix layout (rcr, rrr, ccr, crr)
#   config_name  - Configuration file name without .json extension
#   trait        - Kernel trait combination string
#   tile_config  - Tile configuration parameters
#   config_json  - Full path to JSON configuration file
# ============================================================================
function(create_individual_gemm_test_target datatype layout config_name trait tile_config config_json)
    set(target_name "test_gemm_tile_engine_${datatype}_${layout}_${config_name}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}/${config_name}")
    
    # Generated header path (already created during cmake configuration)
    set(test_header "${working_path}/gemm_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")
    set(test_params_header "${working_path}/test_params.hpp")
    
    # Verify header exists (should have been generated during cmake configuration)
    if(NOT EXISTS ${test_header})
        message(WARNING "Generated header not found: ${test_header}")
        return()
    endif()

    # Verify test parameters header exists
    if(NOT EXISTS ${test_params_header})
        message(WARNING "Test parameters header not found: ${test_params_header}")
        return()
    endif()


    # Create GTest executable for this kernel configuration
    add_gtest_executable(${target_name}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm_simple.cpp
    )

    # Configure GPU architectures for HIP compilation
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_TEST_GPU_TARGETS})

    # Define preprocessor macros for generated header location and test parameters
    target_compile_definitions(${target_name} PRIVATE
        GEMM_SINGLE_INSTANCE_HPP="${test_header}"
        GEMM_TEST_PARAMS_HPP="${test_params_header}"
    )

    # Include directories for headers and dependencies
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
        ${PROJECT_SOURCE_DIR}  # Root directory for tile_engine access
        ${GTEST_INCLUDE_DIRS}
    )

    # Compiler options matching tile_engine requirements
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template  # Suppress template warnings
        -Wno-float-equal              # Allow floating point comparisons
        --offload-compress            # Enable GPU code compression
        -include ${test_header}       # Auto-include generated header
    )

    # Add FP8 format definitions for proper data type interpretation
    if(CK_USE_OCP_FP8)
        target_compile_options(${target_name} PRIVATE -DCK_TILE_USE_OCP_FP8)
    endif()

    message(DEBUG "  Created test target: ${target_name}")
endfunction()

# ============================================================================
# build_gemm_test_targets
#
# Builds all test targets for a specific datatype/layout/config combination.
# Uses tile_engine's two-step process: list kernels, then generate tests.
#
# Parameters:
#   datatype     - Data type (fp16, bf16, fp32, etc.)
#   layout       - Matrix layout (rcr, rrr, ccr, crr)
#   config_name  - Configuration file name without .json extension
# ============================================================================
function(build_gemm_test_targets datatype layout config_name)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}/${config_name}")

    # Locate and validate configuration file
    set(config_filename "${config_name}.json")
    set(json_blob "${CMAKE_CURRENT_SOURCE_DIR}/configs/${config_filename}")

    if(NOT EXISTS ${json_blob})
        message(WARNING "Test config file not found: ${json_blob}")
        return()
    endif()

    # Prepare build directory for this configuration
    file(MAKE_DIRECTORY ${working_path})

    # STEP 1: Discovery phase - list all valid kernel configurations
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --config_json ${json_blob}
                --list_kernels
                --gpu_target "${GEMM_TEST_GPU_TARGETS}"
        WORKING_DIRECTORY ${TILE_ENGINE_GEMM_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(WARNING "Failed to list kernels for ${datatype}_${layout}_${config_name}: ${list_error}")
        return()
    endif()

    # Verify kernel list file was generated
    if(NOT EXISTS ${working_path}/gemm_kernel_list.txt)
        message(DEBUG "No kernels found for ${datatype}_${layout}_${config_name} (validation filtered out all combinations)")
        return()
    endif()

    message(DEBUG "Building tests for ${datatype}_${layout}_${config_name}")

    # STEP 2a: Extract test parameters from config
    set(test_params_file "${working_path}/test_params.hpp")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_SOURCE_DIR}/extract_test_params.py
                --config_file ${json_blob}
                --output_file ${test_params_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE extract_ret
        OUTPUT_VARIABLE extract_output
        ERROR_VARIABLE extract_error
    )

    if(NOT extract_ret EQUAL 0)
        message(WARNING "Failed to extract test parameters for ${datatype}_${layout}: ${extract_error}")
        return()
    endif()

    # STEP 2b: Header generation phase - generate headers using --gen_single
    message(STATUS "  Generating headers using --gen_single...")
    
    file(STRINGS ${working_path}/gemm_kernel_list.txt kernel_lines)
    set(gen_count 0)
    
    foreach(line IN LISTS kernel_lines)
                # Parse kernel specification format: kernel_name|tile_config|trait_combo
                string(REPLACE "|" ";" parts "${line}")
                list(LENGTH parts parts_len)
                if(parts_len EQUAL 3)
                    list(GET parts 0 kernel_name)
                    list(GET parts 1 tile_config)
                    list(GET parts 2 trait_combo)
                    
                    # Generate header using --gen_single
                    execute_process(
                        COMMAND ${Python3_EXECUTABLE} -u ${TILE_ENGINE_GEMM_DIR}/gemm_instance_builder.py
                                --working_path ${working_path}
                                --gpu_target "${GEMM_TEST_GPU_TARGETS}"
                                --datatype ${datatype}
                                --layout ${layout}
                                --config_json ${json_blob}
                                --gen_single
                                --kernel_name "${kernel_name}"
                                --tile_config "${tile_config}"
                                --trait_combo "${trait_combo}"
                        WORKING_DIRECTORY ${TILE_ENGINE_GEMM_DIR}
                        RESULT_VARIABLE gen_ret
                        OUTPUT_VARIABLE gen_output
                        ERROR_VARIABLE gen_error
                    )
                    
                    if(NOT gen_ret EQUAL 0)
                        message(WARNING "Failed to generate header for ${kernel_name}: ${gen_error}")
                    else()
                    math(EXPR gen_count "${gen_count} + 1")
                endif()
            endif()
        endforeach()
        
        message(STATUS "  Generated ${gen_count} headers for ${datatype}_${layout}")

    # STEP 3: Target creation phase - create test targets
    message(STATUS "  Creating test targets...")
    file(STRINGS ${working_path}/gemm_kernel_list.txt kernel_lines)
    set(test_count 0)
    foreach(line IN LISTS kernel_lines)
                # Parse kernel specification format: kernel_name|tile_config|trait_combo
                string(REPLACE "|" ";" parts "${line}")
                list(LENGTH parts parts_len)
                if(parts_len EQUAL 3)
                    list(GET parts 0 kernel_name)
                    list(GET parts 1 tile_config)
                    list(GET parts 2 trait_combo)

                # Generate test target for this kernel configuration
                create_individual_gemm_test_target("${datatype}" "${layout}" "${config_name}" "${trait_combo}" "${tile_config}" "${json_blob}")
                math(EXPR test_count "${test_count} + 1")
            endif()
        endforeach()
        message(STATUS "  Created ${test_count} test targets for ${datatype}_${layout}")
endfunction()# ============================================================================
# MAIN EXECUTION - Test Target Generation
# ============================================================================

message(STATUS "=== Starting GEMM Tile Engine Test Configuration ===")
message(STATUS "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# GPU architecture filtering - only build tests for supported architectures
set(GEMM_TEST_GPU_TARGETS "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950;gfx1201")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_TEST_GPU_TARGETS ${target})
        message(STATUS "  Adding GPU target for tests: ${target}")
    endif()
endforeach()

# Early exit if no compatible GPU architectures are available
if(NOT GEMM_TEST_GPU_TARGETS)
    message(WARNING "Skipping GEMM Tile Engine tests: No supported GPU targets (gfx90a, gfx942, gfx950, gfx1201) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
    return()
endif()

message(STATUS "Building GEMM tile engine tests for GPU targets: ${GEMM_TEST_GPU_TARGETS}")

    # Enable parallel compilation optimizations
    # Set up job pools for better parallel compilation control
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4    # Limit heavy compilations to prevent OOM
        compile_normal=16  # Allow more parallel normal compilations
    )

    # Enable compiler cache if available and explicitly requested
    # Disabled by default due to permission issues in CI environments
    option(ENABLE_CCACHE_TESTS "Enable ccache for test compilation" OFF)
    if(ENABLE_CCACHE_TESTS)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(STATUS "Using ccache for faster test compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(STATUS "ccache disabled for tests (use -DENABLE_CCACHE_TESTS=ON to enable)")
    endif()

# ============================================================================
# Test Configuration Matrix - Clean Focused Design
# ============================================================================

# All supported data types and layouts for comprehensive testing
# Note: fp64 not included (no MFMA hardware support)
set(TEST_DATATYPES "fp16;fp8;bf16;fp32")
set(TEST_LAYOUTS "rcr;rrr;ccr;crr")

# ============================================================================
# Test Target Generation - Datatype-Specific Categories
# ============================================================================

# 1. SMALL DATATYPES: Test optimized config for small data types (fp8, fp16, bf16)
#    These data types can use larger warp tiles due to smaller memory footprint
set(SMALL_DATATYPE_CONFIG "small_datatype_config")
set(SMALL_DATATYPE_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${SMALL_DATATYPE_CONFIG}.json")
set(SMALL_DATATYPES "fp8;fp16;bf16")

if(EXISTS ${SMALL_DATATYPE_CONFIG_FILE})
    message(STATUS "Processing small datatype config: ${SMALL_DATATYPE_CONFIG} (fp8, fp16, bf16)")
    foreach(datatype IN LISTS SMALL_DATATYPES)
        # fp8, fp16, bf16: testing all layouts (rcr, rrr, ccr, crr)
        foreach(layout IN LISTS TEST_LAYOUTS)
            build_gemm_test_targets("${datatype}" "${layout}" "${SMALL_DATATYPE_CONFIG}")
        endforeach()
    endforeach()
else()
    message(WARNING "Small datatype config file not found: ${SMALL_DATATYPE_CONFIG_FILE}")
endif()

# 2. PADDING COVERAGE: Test padding combinations with fixed fp16/rcr configuration
#    This focuses on padding behavior (pad_m, pad_n, pad_k)
set(PADDING_CONFIG "padding_coverage_config")
set(PADDING_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${PADDING_CONFIG}.json")

if(EXISTS ${PADDING_CONFIG_FILE})
    message(STATUS "Processing padding config: ${PADDING_CONFIG} (fp16/rcr only)")
    build_gemm_test_targets("fp16" "rcr" "${PADDING_CONFIG}")
else()
    message(WARNING "Padding config file not found: ${PADDING_CONFIG_FILE}")
endif()

# 3. COVERAGE LEVEL: Quick or comprehensive testing
#    Quick: ~144 kernels with multiple tile sizes and trait combinations
#    Comprehensive: Several thousand kernels with extensive tile sizes, warp configurations, and all trait combinations
set(COVERAGE_LEVEL "quick" CACHE STRING "Coverage level: quick or comprehensive")
set_property(CACHE COVERAGE_LEVEL PROPERTY STRINGS "quick" "comprehensive")

if(COVERAGE_LEVEL STREQUAL "quick")
    set(COVERAGE_CONFIG "quick_coverage_config")
    set(COVERAGE_DESC "Quick - approximately 144 kernels with trait combinations")
elseif(COVERAGE_LEVEL STREQUAL "comprehensive")
    set(COVERAGE_CONFIG "comprehensive_coverage_config")
    set(COVERAGE_DESC "Comprehensive - several thousand kernels with extensive tile and trait coverage")
else()
    message(FATAL_ERROR "Invalid COVERAGE_LEVEL: ${COVERAGE_LEVEL}. Must be 'quick' or 'comprehensive'")
endif()

set(COVERAGE_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/configs/${COVERAGE_CONFIG}.json")

if(EXISTS ${COVERAGE_CONFIG_FILE})
    message(STATUS "Processing coverage config: ${COVERAGE_LEVEL} - ${COVERAGE_DESC}")
    build_gemm_test_targets("fp16" "rcr" "${COVERAGE_CONFIG}")
else()
    message(WARNING "Coverage config file not found: ${COVERAGE_CONFIG_FILE}")
endif()
# ============================================================================


message(STATUS "GEMM tile engine tests configured with datatype-specific design:")
message(STATUS "  - Small datatypes: fp8/fp16/bf16 (all layouts)")
message(STATUS "  - Padding coverage with fp16/rcr")
message(STATUS "  - Coverage level: ${COVERAGE_LEVEL} (~144 kernels quick, several thousand comprehensive)")
message(STATUS "    Use -DCOVERAGE_LEVEL=comprehensive for extensive testing")
