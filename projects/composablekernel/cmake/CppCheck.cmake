# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(CMakeParseArguments)
include(ProcessorCount)
include(Analyzers)

find_program(CPPCHECK_EXE 
    NAMES 
        cppcheck
    PATHS
        /opt/rocm/bin
)

ProcessorCount(CPPCHECK_JOBS)

set(CPPCHECK_BUILD_DIR ${CMAKE_BINARY_DIR}/cppcheck-build)
file(MAKE_DIRECTORY ${CPPCHECK_BUILD_DIR})
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CPPCHECK_BUILD_DIR})

macro(enable_cppcheck)
    set(options FORCE)
    set(oneValueArgs)
    set(multiValueArgs CHECKS SUPPRESS DEFINE UNDEFINE INCLUDE SOURCES)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    string(REPLACE ";" "," CPPCHECK_CHECKS "${PARSE_CHECKS}")
    string(REPLACE ";" "\n" CPPCHECK_SUPPRESS "${PARSE_SUPPRESS};*:/usr/*")
    file(WRITE ${CMAKE_BINARY_DIR}/cppcheck-supressions "${CPPCHECK_SUPPRESS}")
    set(CPPCHECK_DEFINES)
    foreach(DEF ${PARSE_DEFINE})
        set(CPPCHECK_DEFINES "${CPPCHECK_DEFINES} -D${DEF}")
    endforeach()

    set(CPPCHECK_UNDEFINES)
    foreach(DEF ${PARSE_UNDEFINE})
        set(CPPCHECK_UNDEFINES "${CPPCHECK_UNDEFINES} -U${DEF}")
    endforeach()

    set(CPPCHECK_INCLUDES)
    foreach(INC ${PARSE_INCLUDE})
        set(CPPCHECK_INCLUDES "${CPPCHECK_INCLUDES} -I${INC}")
    endforeach()

    # set(CPPCHECK_FORCE)
    set(CPPCHECK_FORCE "--project=${CMAKE_BINARY_DIR}/compile_commands.json")
    if(PARSE_FORCE)
        set(CPPCHECK_FORCE --force)
    endif()

    set(SOURCES)
    set(GLOBS)
    foreach(SOURCE ${PARSE_SOURCES})
        get_filename_component(ABS_SOURCE ${SOURCE} ABSOLUTE)
        if(EXISTS ${ABS_SOURCE})
            if(IS_DIRECTORY ${ABS_SOURCE})
                set(GLOBS "${GLOBS} ${ABS_SOURCE}/*.cpp ${ABS_SOURCE}/*.hpp ${ABS_SOURCE}/*.cxx ${ABS_SOURCE}/*.c ${ABS_SOURCE}/*.h")
            else()
                set(SOURCES "${SOURCES} ${ABS_SOURCE}")
            endif()
        else()
            set(GLOBS "${GLOBS} ${ABS_SOURCE}")
        endif()
    endforeach()

    file(WRITE ${CMAKE_BINARY_DIR}/cppcheck.cmake "
        file(GLOB_RECURSE GSRCS ${GLOBS})
        set(CPPCHECK_COMMAND
            ${CPPCHECK_EXE}
            -q
            # -v
            # --report-progress
            ${CPPCHECK_FORCE}
            --cppcheck-build-dir=${CPPCHECK_BUILD_DIR}
            --platform=native
            --template=gcc
            --error-exitcode=1
            -j ${CPPCHECK_JOBS}
            ${CPPCHECK_DEFINES}
            ${CPPCHECK_UNDEFINES}
            ${CPPCHECK_INCLUDES}
            --enable=${CPPCHECK_CHECKS}
            --inline-suppr
            --suppressions-list=${CMAKE_BINARY_DIR}/cppcheck-supressions
            ${SOURCES} \${GSRCS}
        )
        string(REPLACE \";\" \" \" CPPCHECK_SHOW_COMMAND \"\${CPPCHECK_COMMAND}\")
        message(\"\${CPPCHECK_SHOW_COMMAND}\")
        execute_process(
            COMMAND \${CPPCHECK_COMMAND}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE RESULT
        )
        if(NOT RESULT EQUAL 0)
            message(FATAL_ERROR \"Cppcheck failed\")
        endif()
")

    add_custom_target(cppcheck
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cppcheck.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "cppcheck: Running cppcheck..."
    )
    mark_as_analyzer(cppcheck)
endmacro()


