# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT


function(generate_sharded_instantiations)
    cmake_parse_arguments(
        GEN_SHARDED
        # No boolean arguments
        ""
        # Single-value arguments
        "INSTANCES_NAME;TEMPLATE_FILE;NUM_SHARDS;OUTPUT_DIR;SRC_LIST"
        # No multi-value arguments.
        ""
        ${ARGN}
    )
    if (NOT GEN_SHARDED_INSTANCES_NAME)
        message(FATAL_ERROR "INSTANCES_NAME is required for generate_sharded_instantiations")
    endif()
    if (NOT GEN_SHARDED_TEMPLATE_FILE)
        message(FATAL_ERROR "TEMPLATE_FILE is required for generate_sharded_instantiations")
    endif()
    if (NOT GEN_SHARDED_NUM_SHARDS)
        message(FATAL_ERROR "NUM_SHARDS is required for generate_sharded_instantiations")
    endif()
    if(NOT GEN_SHARDED_OUTPUT_DIR)
        message(FATAL_ERROR "OUTPUT_DIR is required for generate_sharded_instantiations")
    endif()
    if (NOT GEN_SHARDED_SRC_LIST)
        message(FATAL_ERROR "SRC_LIST is required for generate_sharded_instantiations")
    endif()

    file(MAKE_DIRECTORY ${GEN_SHARDED_OUTPUT_DIR})


    set(GENERATED_SOURCE_FILES "")
    set(EXTERN_TEMPLATE_STATEMENTS "")
    set(CALL_STATEMENTS "")
    message(DEBUG "Generating sharded instantiations for target: ${GEN_SHARDED_INSTANCES_NAME}")

    set(INSTANCES "${GEN_SHARDED_INSTANCES_NAME}")
    
    # Generate the inc file with the template function defintions.
    # This include file will hold the template function definitions and a using alias for all the shard
    # instantiation functions.
    configure_file(
        "${GEN_SHARDED_TEMPLATE_FILE}"
        "${GEN_SHARDED_OUTPUT_DIR}/${INSTANCES}.inc"
        @ONLY
    )

    # Generate the sharded instantiation functions.
    # This is where the build parallelization happens.
    # Each of these source files will contain a single instantiation function for a shard,
    # which will be called sequentially by the caller function.
    set(INC_DIR "${GEN_SHARDED_INC_DIR}")
    math(EXPR LAST_SHARD_ID "${GEN_SHARDED_NUM_SHARDS} - 1")
    foreach(SHARD_ID RANGE 0 ${LAST_SHARD_ID})
        set(NUM_SHARDS "${GEN_SHARDED_NUM_SHARDS}")
        set(SHARD_FUNCTION_PATH "${GEN_SHARDED_OUTPUT_DIR}/${INSTANCES}_shard_${SHARD_ID}.cpp")
        set(SHARD_FUNCTION_TEMPLATE "${PROJECT_SOURCE_DIR}/cmake/instantiate_shard.in")
        configure_file(
            "${SHARD_FUNCTION_TEMPLATE}"
            "${SHARD_FUNCTION_PATH}"
            @ONLY
        )
        list(APPEND GENERATED_SOURCE_FILES "${SHARD_FUNCTION_PATH}")
        set(SHARDED_FUNCTION_NAME "add_${INSTANCES}_shard<${NUM_SHARDS}, ${SHARD_ID}>")
        list(APPEND EXTERN_TEMPLATE_STATEMENTS "extern template void\n${SHARDED_FUNCTION_NAME}(\n  ${INSTANCES}& instances)")
        list(APPEND CALL_STATEMENTS "  ${SHARDED_FUNCTION_NAME}(instances)")
    endforeach()

    # Join the include statements, the extern template declarations, and the call statements each
    # into a single string for variable substitution in the caller function.
    string(REPLACE ";" ";\n" INCLUDE_STATEMENTS "${INCLUDE_STATEMENTS}")
    string(REPLACE ";" ";\n" CALL_STATEMENTS "${CALL_STATEMENTS}")
    string(REPLACE ";" ";\n" EXTERN_TEMPLATE_STATEMENTS "${EXTERN_TEMPLATE_STATEMENTS}")

    # Generate the caller function.
    set(CALLER_FUNCTION_PATH "${GEN_SHARDED_OUTPUT_DIR}/${INSTANCES}.cpp")
    set(FUNCTION_TEMPLATE "${PROJECT_SOURCE_DIR}/cmake/call_shard.in")
    configure_file(
        "${FUNCTION_TEMPLATE}"
        "${CALLER_FUNCTION_PATH}"
        @ONLY
    )
    list(APPEND GENERATED_SOURCE_FILES "${CALLER_FUNCTION_PATH}")

    # Add the generated source files to the list of source files.
    # This allows the generated source files to be included in the build.
    list(APPEND ${GEN_SHARDED_SRC_LIST} ${GENERATED_SOURCE_FILES})
    set(${GEN_SHARDED_SRC_LIST} "${${GEN_SHARDED_SRC_LIST}}" PARENT_SCOPE)
endfunction()