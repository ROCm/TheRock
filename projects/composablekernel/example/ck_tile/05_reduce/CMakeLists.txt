# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(EXAMPLE_REDUCE "tile_example_reduce")
# not using add_example_executable() to add this target, since we don't want this to have
# to be included in "make all/install/check"
message(DEBUG "adding example ${EXAMPLE_REDUCE}")

add_executable(${EXAMPLE_REDUCE} reduce.cpp)
target_include_directories(${EXAMPLE_REDUCE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_REDUCE_COMPILE_OPTIONS)

# NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
list(APPEND EXAMPLE_REDUCE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)

target_compile_options(${EXAMPLE_REDUCE} PRIVATE ${EXAMPLE_REDUCE_COMPILE_OPTIONS})

# Multi Reduce Threadwise Example
set(EXAMPLE_MULTI_REDUCE "tile_example_multi_reduce_threadwise")
add_executable(${EXAMPLE_MULTI_REDUCE} EXCLUDE_FROM_ALL multiple_reduce_threadwise.cpp)
target_include_directories(${EXAMPLE_MULTI_REDUCE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_MULTI_REDUCE_COMPILE_OPTIONS)
list(APPEND EXAMPLE_MULTI_REDUCE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
target_compile_options(${EXAMPLE_MULTI_REDUCE} PRIVATE ${EXAMPLE_MULTI_REDUCE_COMPILE_OPTIONS})

# Multi Reduce Blockwise Example
set(EXAMPLE_MULTI_REDUCE_BLOCKWISE "tile_example_multi_reduce_multiblock")
add_executable(${EXAMPLE_MULTI_REDUCE_BLOCKWISE} EXCLUDE_FROM_ALL multiple_reduce_multiblock.cpp)
target_include_directories(${EXAMPLE_MULTI_REDUCE_BLOCKWISE} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
set(EXAMPLE_MULTI_REDUCE_BLOCKWISE_COMPILE_OPTIONS)
list(APPEND EXAMPLE_MULTI_REDUCE_BLOCKWISE_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
target_compile_options(${EXAMPLE_MULTI_REDUCE_BLOCKWISE} PRIVATE ${EXAMPLE_MULTI_REDUCE_BLOCKWISE_COMPILE_OPTIONS})

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
