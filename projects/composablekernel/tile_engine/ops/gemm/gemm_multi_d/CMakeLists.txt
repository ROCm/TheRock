# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(GEMM_MULTI_D_DATATYPE "fp16" CACHE STRING "List of datatypes for GEMM Multi D (semicolon-separated)")
set(GEMM_MULTI_D_LAYOUT "rcrr;rrrr;crrr;ccrr" CACHE STRING "List of layout for GEMM Multi D (semicolon-separated)")
set(GEMM_MULTI_D_CONFIG_FILE "" CACHE STRING "Custom config file name (without path, must be in configs/ folder)")
set(GEMM_MULTI_D_ELEMENTWISE_FUNCTION "mul"  CACHE STRING "Elementwise function")

option(ENABLE_CCACHE_GEMM_MULTI_D "Enable ccache for GEMM Multi D ops compilation" OFF)

# Store the directory path for use in functions
set(GEMM_MULTI_D_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Function to create individual GEMM Multi D targets
function(create_individual_gemm_multi_d_target datatype layout trait tile_config config_json)
    # Use the parent scope GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL variable
    if(NOT GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL)
        message(WARNING "Skipping individual GEMM Multi D target ${datatype}_${layout}_${trait}_${tile_config}: No supported GPU targets")
        return()
    endif()

    # Parse tile configuration: format is tile_mxtile_nxtile_k_warp_mxwarp_nxwarp_k_warp_tile_mxwarp_tile_nxwarp_tile_k
    # First split by underscore to get three groups
    string(REPLACE "_" ";" config_groups ${tile_config})
    list(GET config_groups 0 tile_dims)      # e.g., 256x256x32
    list(GET config_groups 1 warp_dims)      # e.g., 4x1x1
    list(GET config_groups 2 warp_tile_dims) # e.g., 16x16x16

    # Parse tile dimensions
    string(REPLACE "x" ";" tile_parts ${tile_dims})
    list(GET tile_parts 0 tile_m)
    list(GET tile_parts 1 tile_n)
    list(GET tile_parts 2 tile_k)

    # Parse warp dimensions
    string(REPLACE "x" ";" warp_parts ${warp_dims})
    list(GET warp_parts 0 warp_m)
    list(GET warp_parts 1 warp_n)
    list(GET warp_parts 2 warp_k)

    # Parse warp tile dimensions
    string(REPLACE "x" ";" warp_tile_parts ${warp_tile_dims})
    list(GET warp_tile_parts 0 warp_tile_m)
    list(GET warp_tile_parts 1 warp_tile_n)
    list(GET warp_tile_parts 2 warp_tile_k)

    set(target_name "benchmark_gemm_multi_d_${datatype}_${layout}_${trait}_${tile_config}")
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Generate the single instance header for this kernel
    set(instance_header "${working_path}/gemm_multi_d_single_${datatype}_${layout}_${trait}_${tile_config}.hpp")

    # Add custom command to generate the header file at build time
    add_custom_command(
        OUTPUT ${instance_header}
        COMMAND ${Python3_EXECUTABLE} ${GEMM_MULTI_D_SOURCE_DIR}/gemm_multi_d_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --elementwise_function ${GEMM_MULTI_D_ELEMENTWISE_FUNCTION}
                --config_json ${config_json}
                --gen_single
                --kernel_name "gemm_multi_d_${datatype}_${layout}_${trait}_${tile_config}"
                --tile_config "${tile_config}"
                --trait_combo "${trait}"
                --gpu_target "${GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL}"
        DEPENDS ${GEMM_MULTI_D_SOURCE_DIR}/gemm_multi_d_instance_builder.py ${config_json}
        COMMENT "Generating ${instance_header}"
    )

    # Create the executable
    add_executable(${target_name}
        EXCLUDE_FROM_ALL
        ${GEMM_MULTI_D_SOURCE_DIR}/gemm_multi_d_benchmark_single.cpp
        ${instance_header}
    )

    # Set GPU architectures
    set_property(TARGET ${target_name} PROPERTY HIP_ARCHITECTURES ${GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL})

    # Set compile definitions
    target_compile_definitions(${target_name} PRIVATE
        GEMM_MULTI_D_SINGLE_INSTANCE_HPP="${instance_header}"
    )

    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${GEMM_MULTI_D_SOURCE_DIR}
        ${working_path}
    )

    # Compile options
    target_compile_options(${target_name} PRIVATE
        -Wno-undefined-func-template
        -Wno-float-equal
        --offload-compress
        -include ${instance_header}
    )

    # Add to collection targets
    add_dependencies(benchmark_gemm_multi_d_all ${target_name})
    add_dependencies(benchmark_gemm_multi_d_${datatype} ${target_name})
    add_dependencies(benchmark_gemm_multi_d_${layout} ${target_name})
    add_dependencies(benchmark_gemm_multi_d_${datatype}_${layout} ${target_name})

    # Add to trait-specific targets
    string(REPLACE "_" ";" trait_parts ${trait})
    list(GET trait_parts 0 pipeline)
    list(GET trait_parts 1 epilogue)
    list(GET trait_parts 2 scheduler)

    add_dependencies(benchmark_gemm_multi_d_${pipeline}_pipeline ${target_name})
    add_dependencies(benchmark_gemm_multi_d_${epilogue}_epilogue ${target_name})
    add_dependencies(benchmark_gemm_multi_d_${scheduler}_scheduler ${target_name})
endfunction()

# Function to build individual GEMM Multi D targets
function(build_individual_gemm_multi_d_targets datatype layout)
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${layout}")

    # Choose config file
    # Priority order:
    # 1. Environment variable GEMM_MULTI_D_CONFIG_FILE
    # 2. CMake variable GEMM_MULTI_D_CONFIG_FILE
    # 3. Default based on layout

    # Check environment variable first
    if(DEFINED ENV{GEMM_MULTI_D_CONFIG_FILE} AND NOT "$ENV{GEMM_MULTI_D_CONFIG_FILE}" STREQUAL "")
        set(config_filename "$ENV{GEMM_MULTI_D_CONFIG_FILE}")
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${config_filename}")
        message(VERBOSE "  Using config from environment variable: ${config_filename}")
    elseif(NOT "${GEMM_MULTI_D_CONFIG_FILE}" STREQUAL "")
        # Use CMake variable if set
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/${GEMM_MULTI_D_CONFIG_FILE}")
        message(VERBOSE "  Using custom config: ${GEMM_MULTI_D_CONFIG_FILE}")
    else()
        # Use default config for all layouts
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_config.json")
        message(VERBOSE "  Using default config for layout ${layout}")
    endif()

    # Check if config file exists
    if(NOT EXISTS ${json_blob})
        message(FATAL_ERROR "Config file not found: ${json_blob}")
    endif()

    # Determine number of workers for parallel generation
    if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
        set(num_workers $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    else()
        # Use processor count but limit to avoid memory issues
        cmake_host_system_information(RESULT num_cores QUERY NUMBER_OF_LOGICAL_CORES)
        math(EXPR num_workers "${num_cores}")
        if(num_workers GREATER 8)
            set(num_workers 8)
        endif()
    endif()

    # Generate individual kernel files using parallel version
    message(VERBOSE "Generating individual kernels for ${datatype} ${layout} using ${num_workers} workers...")
    message(VERBOSE "  Working path: ${working_path}")
    message(VERBOSE "  Config file: ${json_blob}")
    message(VERBOSE "  Python executable: ${Python3_EXECUTABLE}")
    message(VERBOSE "  Script path: ${CMAKE_CURRENT_LIST_DIR}/gemm_multi_d_instance_builder.py")

    # Create working directory first
    file(MAKE_DIRECTORY ${working_path})

    message(VERBOSE "COMMAND: ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_multi_d_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --elementwise_function ${GEMM_MULTI_D_ELEMENTWISE_FUNCTION}
                --config_json ${json_blob}
                --gpu_target ${GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL}
                --list_kernels ")

    # First, just list the kernels (fast operation)
    message(VERBOSE "  Listing kernel configurations...")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -u ${CMAKE_CURRENT_LIST_DIR}/gemm_multi_d_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --layout ${layout}
                --elementwise_function ${GEMM_MULTI_D_ELEMENTWISE_FUNCTION}
                --config_json ${json_blob}
                --gpu_target ${GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL}
                --list_kernels  
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        RESULT_VARIABLE ret
        OUTPUT_VARIABLE list_output
        ERROR_VARIABLE list_error
    )

    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${layout}: ${list_error}")
    endif()

    # Read kernel count
    if(EXISTS ${working_path}/gemm_multi_d_kernel_count.txt)
        file(READ ${working_path}/gemm_multi_d_kernel_count.txt kernel_count)
        string(STRIP "${kernel_count}" kernel_count)
        message(VERBOSE "  Found ${kernel_count} kernel configurations")
    else()
        message(FATAL_ERROR "Kernel count file not found")
    endif()

    # Read kernel list and create targets
    if(EXISTS ${working_path}/gemm_multi_d_kernel_list.txt)
        file(STRINGS ${working_path}/gemm_multi_d_kernel_list.txt kernel_lines)
        foreach(line IN LISTS kernel_lines)
            # Parse line: kernel_name|tile_config|trait_combo
            string(REPLACE "|" ";" parts "${line}")
            list(GET parts 0 kernel_name)
            list(GET parts 1 tile_config)
            list(GET parts 2 trait_combo)

            # Create individual target
            create_individual_gemm_multi_d_target("${datatype}" "${layout}" "${trait_combo}" "${tile_config}" "${json_blob}")
        endforeach()
    else()
        message(FATAL_ERROR "Kernel list file not found")
    endif()
endfunction()

# Main build logic - Only individual builds supported
message(VERBOSE "=== Starting Tile Engine GEMM Multi D Configuration ===")
message(VERBOSE "GEMM_MULTI_D_DATATYPE: ${GEMM_MULTI_D_DATATYPE}")
message(VERBOSE "GEMM_MULTI_D_LAYOUT: ${GEMM_MULTI_D_LAYOUT}")
message(VERBOSE "SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")

# Filter GPU targets to only gfx90a, gfx942, gfx950
set(GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL "")
set(DESIRED_TARGETS "gfx90a;gfx942;gfx950")

foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
    if(target IN_LIST DESIRED_TARGETS)
        list(APPEND GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL ${target})
        message(VERBOSE "  Adding GPU target: ${target}")
    endif()
endforeach()

# Skip build if no matching targets found
if(NOT GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL)
    message(WARNING "Skipping Tile Engine GEMM Multi D build: No supported GPU targets (gfx90a, gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
else()
    message(VERBOSE "Building individual GEMM Multi D targets for GPU targets: ${GEMM_MULTI_D_GPU_TARGETS_INDIVIDUAL}")

    # Enable parallel compilation optimizations
    # Set up job pools for better parallel compilation control
    set_property(GLOBAL PROPERTY JOB_POOLS
        compile_heavy=4    # Limit heavy compilations to prevent OOM
        compile_normal=16  # Allow more parallel normal compilations
    )

    # Enable compiler cache if available and explicitly requested
    # Disabled by default due to permission issues in CI environments
    if(ENABLE_CCACHE_GEMM_MULTI_D)
        find_program(CCACHE_PROGRAM ccache)
        if(CCACHE_PROGRAM)
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
            message(VERBOSE "Using ccache for faster compilation")
        else()
            message(WARNING "ccache requested but not found")
        endif()
    else()
        message(VERBOSE "ccache disabled for GEMM Multi D ops (use -DENABLE_CCACHE_GEMM_MULTI_D=ON to enable)")
    endif()

    # Create master collection targets
    add_custom_target(benchmark_gemm_multi_d_all)

    # Create datatype collection targets
    foreach(dt IN LISTS GEMM_MULTI_D_DATATYPE)
        add_custom_target(benchmark_gemm_multi_d_${dt})
    endforeach()

    # Create layout collection targets
    foreach(l IN LISTS GEMM_MULTI_D_LAYOUT)
        add_custom_target(benchmark_gemm_multi_d_${l})
    endforeach()

    # Create combined collection targets
    foreach(dt IN LISTS GEMM_MULTI_D_DATATYPE)
        foreach(l IN LISTS GEMM_MULTI_D_LAYOUT)
            add_custom_target(benchmark_gemm_multi_d_${dt}_${l})
        endforeach()
    endforeach()

    # Create trait-based collection targets
    # These are common trait components used across all GEMM Multi D kernels
    set(GEMM_MULTI_D_PIPELINES "mem;compv3;compv4")
    set(GEMM_MULTI_D_EPILOGUES "default;cshuffle")
    set(GEMM_MULTI_D_SCHEDULERS "intrawave;interwave")

    foreach(pipeline IN LISTS GEMM_MULTI_D_PIPELINES)
        add_custom_target(benchmark_gemm_multi_d_${pipeline}_pipeline)
    endforeach()

    foreach(epilogue IN LISTS GEMM_MULTI_D_EPILOGUES)
        add_custom_target(benchmark_gemm_multi_d_${epilogue}_epilogue)
    endforeach()

    foreach(scheduler IN LISTS GEMM_MULTI_D_SCHEDULERS)
        add_custom_target(benchmark_gemm_multi_d_${scheduler}_scheduler)
    endforeach()

    # Build individual targets for each datatype/layout combination
    foreach(dt IN LISTS GEMM_MULTI_D_DATATYPE)
        foreach(l IN LISTS GEMM_MULTI_D_LAYOUT)
            build_individual_gemm_multi_d_targets(${dt} ${l})
        endforeach()
    endforeach()
endif()
