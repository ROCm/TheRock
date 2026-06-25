# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function (add_moe_smoothquant_example TARGET_NAME MAIN_SRC)
    message(DEBUG "adding ${TARGET_NAME}")
    # not using add_example_executable() to add target, since we don't want this to have
    # to be included in "make all/install/check"
    add_executable(${TARGET_NAME} ${MAIN_SRC})
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR})

    foreach(source IN LISTS ARGN)
        list(APPEND INSTANCE_SRCS ${source})
    endforeach()

    target_sources(${TARGET_NAME} PRIVATE ${INSTANCE_SRCS})

    set(COMPILE_OPTIONS)
    # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
    list(APPEND COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
    # list(APPEND COMPILE_OPTIONS -v --save-temps -Wno-gnu-line-marker)

    target_compile_options(${TARGET_NAME} PRIVATE ${COMPILE_OPTIONS})
endfunction(add_moe_smoothquant_example TARGET_NAME MAIN_SRC)

file(GLOB INSTANCE_SRCS instances/*.cpp)

add_moe_smoothquant_example(tile_example_moe_smoothquant moe_smoothquant.cpp ${INSTANCE_SRCS})
