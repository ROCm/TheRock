# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(create_tile_add_rmsnorm2d_rdquant_fwd SUFFIX)
    set(TILE_ADD_RMSNORM2D_RDQUANT_FWD "test_ck_tile_add_rmsnorm2d_rdquant_fwd_${SUFFIX}")
    message(DEBUG "adding ${TILE_ADD_RMSNORM2D_RDQUANT_FWD}")
    file(GLOB INSTANCE_SRCS instances/*.cpp)
    add_test_executable(${TILE_ADD_RMSNORM2D_RDQUANT_FWD} add_rmsnorm2d_rdquant_fwd_${SUFFIX}.cpp)
    target_include_directories(${TILE_ADD_RMSNORM2D_RDQUANT_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    target_sources(${TILE_ADD_RMSNORM2D_RDQUANT_FWD} PRIVATE ${INSTANCE_SRCS})

    set(TILE_ADD_RMSNORM2D_RDQUANT_FWD_COMPILE_OPTIONS)
    # NOTE: we turn off undefined-func-template to let source compile without explicit declare function specializations
    list(APPEND TILE_ADD_RMSNORM2D_RDQUANT_FWD_COMPILE_OPTIONS -Wno-undefined-func-template -Wno-float-equal)
    target_compile_options(${TILE_ADD_RMSNORM2D_RDQUANT_FWD} PRIVATE ${TILE_ADD_RMSNORM2D_RDQUANT_FWD_COMPILE_OPTIONS})

    # TODO: we have to turn off this global prop, otherwise the progress bar generated
    # by cmake will print too many files, execvp: /bin/sh: Argument list too long
    # however, this property may affect global
    # TODO: consider codegen a makefile by us
    set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
endfunction()

if(GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
    create_tile_add_rmsnorm2d_rdquant_fwd("fp16")
    create_tile_add_rmsnorm2d_rdquant_fwd("bf16")
else()
    message(DEBUG "Skipping ck tile add_rmsnorm2d_rdquant_fwd tests for current target")
endif()
