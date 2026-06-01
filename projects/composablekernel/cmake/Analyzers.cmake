# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT TARGET analyze)
    add_custom_target(analyze)
endif()

function(mark_as_analyzer)
    add_dependencies(analyze ${ARGN})
endfunction()

