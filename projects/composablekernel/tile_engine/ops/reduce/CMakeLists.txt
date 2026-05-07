# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# cmake_minimum_required(VERSION 4.2)

# enable_testing()

set(MULTI_REDUCE_DATATYPE "fp16" CACHE STRING "List of datatypes Multi Reduce (semicolon-separated)")
set(MULTI_REDUCE_VARIANTS "multiops_multiblock;multiops_threadwise" CACHE STRING "List of variants for Multi Reduce (semicolon-separated)")

function(build_multi_reduce_for_datatype datatype variant)
    # Filter GPU targets to only gfx942, and gfx950
    set(GPU_TARGETS "")
    set(DESIRED_TARGETS "gfx942;gfx950")
    set(VALID_VARIANTS "multiops_multiblock;multiops_threadwise")
    
    foreach(target IN LISTS SUPPORTED_GPU_TARGETS)
        if(target IN_LIST DESIRED_TARGETS)
            list(APPEND GPU_TARGETS ${target})
        endif()
    endforeach()
    
    # Skip compilation if no matching targets found
    if(NOT GPU_TARGETS)
        message(WARNING "Skipping Tile Engine for Multi Reduction Kernel: No supported GPU targets (gfx942, gfx950) found in SUPPORTED_GPU_TARGETS: ${SUPPORTED_GPU_TARGETS}")
        return()
    endif()
    
    message(STATUS "Building Reduction for GPU targets: ${GPU_TARGETS}")
    
    set(working_path "${CMAKE_CURRENT_BINARY_DIR}/${datatype}/${variant}")
    file(MAKE_DIRECTORY "${working_path}")

    # Comment this if-else block when using user_provided_config
    if(variant IN_LIST VALID_VARIANTS)
        set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/default_multi_reduce_config.json")
    else()
        # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/custom_ci_config.json")
        message(WARNING "Unknown Multi Reduce variant: ${variant}.")
        return()
    endif()

    # uncomment this if you want to use user_provided_config.json
    # set(json_blob "${CMAKE_CURRENT_LIST_DIR}/configs/user_provided_config.json")
    
    # Generate kernel list
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/reduce_instance_builder.py
                --working_path ${working_path}
                --datatype ${datatype}
                --variant ${variant}
                --config_json ${json_blob}
                --list_blobs
                --gpu_target "${GPU_TARGETS}"
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to list kernels for ${datatype} ${variant}: ${ret}")
    endif()

    file(STRINGS "${working_path}/reduce_${variant}_blobs_list.txt" codegen_blobs)
    
    # Generate the blobs
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/reduce_instance_builder.py
                --working_path "${working_path}"
                --datatype ${datatype}
                --config_json "${json_blob}"
                --variant "${variant}"
                --gen_blobs
                --gpu_target "${GPU_TARGETS}"
        RESULT_VARIABLE ret
    )
    if(NOT ret EQUAL 0)
        message(FATAL_ERROR "Failed to generate kernels for ${datatype} ${variant}: ${ret}")
    endif()

    message(STATUS "Generated ${datatype} ${variant} reduction kernel blobs at: ${working_path}")

    # # Add test executables for each generated test
    file(STRINGS "${working_path}/reduce_${variant}_blobs_list.txt" test_basenames)

    foreach(test_base IN LISTS test_basenames)
        string(PREPEND test_base "test_")
        set(test_src "${working_path}/${test_base}.cpp")
        set(test_target "${test_base}")

        add_executable(${test_target} ${test_src})
        target_include_directories(${test_target} PRIVATE
            "${CMAKE_SOURCE_DIR}/test/ck_tile/reduce/"
            ${working_path}
        )

        target_compile_options(${test_target} PRIVATE -Wno-global-constructors -Wno-dev)
        target_link_libraries(${test_target} PRIVATE gtest gtest_main)

        add_test(NAME ${test_target} COMMAND ${test_target})
        set_tests_properties(${test_target} PROPERTIES LABELS "multi_reduce")
    endforeach()
    add_custom_target(test_reduce_${variant}_${datatype} DEPENDS ${codegen_blobs})

    # # Generating a single binary from all the tests (debug-only)
    # set(test_srcs)
    # foreach(test_base IN LISTS test_basenames)
    #     list(APPEND test_srcs "${working_path}/test_${test_base}.cpp")
    # endforeach()

    # if(test_srcs)
    #     set(test_target "test_reduce_${variant}_${datatype}")
    #     add_executable(${test_target} ${test_srcs})
    #     target_include_directories(${test_target} PRIVATE
    #         ${working_path}
    #         "${CMAKE_SOURCE_DIR}/test/ck_tile/reduce/"
    #     )
    #     target_compile_options(${test_target} PRIVATE -Wno-global-constructors -Wno-dev)
    #     target_link_libraries(${test_target} PRIVATE gtest gtest_main)
    # endif()

endfunction()

# Process each datatype in isolation
foreach(dt IN LISTS MULTI_REDUCE_DATATYPE)
    foreach(l IN LISTS MULTI_REDUCE_VARIANTS)
        build_multi_reduce_for_datatype(${dt} ${l})
    endforeach()
endforeach()