# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(create_tile_layernorm2d_fwd SUFFIX)
    set(TEST_CK_TILE_LAYERNORM2D_FWD "test_ck_tile_layernorm2d_fwd_${SUFFIX}")

    message(DEBUG "adding example ${TEST_CK_TILE_LAYERNORM2D_FWD}")
    add_test_executable(${TEST_CK_TILE_LAYERNORM2D_FWD} layernorm2d_fwd_${SUFFIX}.cpp)
    target_include_directories(${TEST_CK_TILE_LAYERNORM2D_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    target_sources(${TEST_CK_TILE_LAYERNORM2D_FWD} PRIVATE ${LAYERNORM2D_FWD_GEN_BLOBS})

    set(TEST_CK_TILE_LAYERNORM2D_FWD_COMPILE_OPTIONS)

    # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
    list(APPEND TEST_CK_TILE_LAYERNORM2D_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal --offload-compress)

    target_compile_options(${TEST_CK_TILE_LAYERNORM2D_FWD} PRIVATE ${TEST_CK_TILE_LAYERNORM2D_FWD_COMPILE_OPTIONS})
endfunction()

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    set(LAYERNORM2D_FWD_KNOWN_APIS "fwd;bwd")
    set(LAYERNORM2D_FWD_ENABLE_APIS  "fwd" CACHE STRING
        "semicolon-separated list of APIs to generate (${LAYERNORM2D_FWD_KNOWN_APIS}) & link, or \"all\".")
    if(LAYERNORM2D_FWD_ENABLE_APIS  STREQUAL "all")
        set(LAYERNORM2D_FWD_ENABLE_APIS  ${LAYERNORM2D_FWD_KNOWN_APIS})
    endif()

    # generate a list of kernels, but not actually emit files at config sta
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
        --api ${LAYERNORM2D_FWD_ENABLE_APIS} --working_path ${CMAKE_CURRENT_BINARY_DIR} --list_blobs
        RESULT_VARIABLE ret
        )
    if(ret AND NOT ret EQUAL 0)
        message( FATAL_ERROR "Fail to generate kernels via Python. ${ret}")
    endif()

    file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/layernorm2d_fwd_blobs.txt LAYERNORM2D_FWD_GEN_BLOBS)

    add_custom_command(
        OUTPUT ${LAYERNORM2D_FWD_GEN_BLOBS}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/generate.py
        --api ${LAYERNORM2D_FWD_ENABLE_APIS} --working_path ${CMAKE_CURRENT_BINARY_DIR} --gen_blobs
        )

    create_tile_layernorm2d_fwd("fp16")
    create_tile_layernorm2d_fwd("bf16")

# TODO: we have to turn off this global prop, otherwise the progress bar generated
# by cmake will print too many files, execvp: /bin/sh: Argument list too long
# however, this property may affect global
# TODO: consider codegen a makefile by us
    set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
else()
    message(DEBUG "Skipping ck tile add_rmsnorm2d_rdquant_fwd tests for current target")
endif()
