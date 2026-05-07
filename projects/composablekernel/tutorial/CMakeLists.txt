# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_directories(BEFORE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/library/include
)

message(STATUS "Building tutorials...")
add_custom_target(tutorials)

# add all tutorial subdir
file(GLOB dir_list LIST_DIRECTORIES true *)
FOREACH(subdir ${dir_list})
    if(IS_DIRECTORY "${subdir}" AND EXISTS "${subdir}/CMakeLists.txt")
        add_subdirectory(${subdir})
    ENDIF()
ENDFOREACH()
