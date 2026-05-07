# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)
include(FetchContent)

set(GOOGLETEST_DIR "" CACHE STRING "Location of local GoogleTest repo to build against")

if(GOOGLETEST_DIR)
  set(FETCHCONTENT_SOURCE_DIR_GOOGLETEST ${GOOGLETEST_DIR} CACHE STRING "GoogleTest source directory override")
endif()

FetchContent_Declare(
    GTest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571
)

FetchContent_Populate(GTest)

# Patch googlemock/CMakeLists.txt to fix invalid include path
set(GMOCK_CMAKE "${gtest_SOURCE_DIR}/googlemock/CMakeLists.txt")
file(READ "${GMOCK_CMAKE}" GMOCK_CMAKE_CONTENT)
string(REPLACE [[gtest_SOURCE_DIR}/include]]
               [[gtest_SOURCE_DIR}/googletest/include]]
               GMOCK_CMAKE_CONTENT
               "${GMOCK_CMAKE_CONTENT}")
file(WRITE "${GMOCK_CMAKE}" "${GMOCK_CMAKE_CONTENT}")

# Suppress ROCMChecks WARNING on GoogleTests
set(ROCM_DISABLE_CHECKS FALSE)
macro(rocm_check_toolchain_var var access value list_file)
    if(NOT ROCM_DISABLE_CHECKS)
        _rocm_check_toolchain_var("${var}" "${access}" "${value}" "${list_file}")
    endif()
endmacro()

if(WIN32)
    set(gtest_force_shared_crt ON CACHE_INTERNAL "")
endif()

set(BUILD_GMOCK ON CACHE INTERNAL "")
set(INSTALL_GTEST OFF CACHE INTERNAL "")

# Store the current value of BUILD_SHARED_LIBS
set(__build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

set(ROCM_DISABLE_CHECKS TRUE)
add_subdirectory(${gtest_SOURCE_DIR} ${gtest_BINARY_DIR})
set(ROCM_DISABLE_CHECKS FALSE)

# Restore the old value of BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)

set(GTEST_CXX_FLAGS
     -Wno-undef
     -Wno-reserved-identifier
     -Wno-global-constructors
     -Wno-missing-noreturn
     -Wno-disabled-macro-expansion
     -Wno-used-but-marked-unused
     -Wno-switch-enum
     -Wno-zero-as-null-pointer-constant
     -Wno-unused-member-function
     -Wno-comma
     -Wno-old-style-cast
     -Wno-deprecated
     -Wno-unsafe-buffer-usage
     -Wno-float-equal
)

if(WIN32)
    list(APPEND GTEST_CXX_FLAGS
            -Wno-suggest-destructor-override
            -Wno-suggest-override
            -Wno-nonportable-system-include-path
            -Wno-language-extension-token)
endif()

target_compile_options(gtest PRIVATE ${GTEST_CXX_FLAGS})
target_compile_options(gtest_main PRIVATE ${GTEST_CXX_FLAGS})
target_compile_definitions(gtest PRIVATE GTEST_HAS_SEH=0)
target_compile_definitions(gtest_main PRIVATE GTEST_HAS_SEH=0)

if(TARGET gmock)
    target_compile_options(gmock PRIVATE ${GTEST_CXX_FLAGS})
    target_compile_definitions(gmock PRIVATE GTEST_HAS_SEH=0)
endif()

if(TARGET gmock_main)
    target_compile_options(gmock_main PRIVATE ${GTEST_CXX_FLAGS})
    target_compile_definitions(gmock_main PRIVATE GTEST_HAS_SEH=0)
endif()
