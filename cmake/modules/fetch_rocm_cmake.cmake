# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(ROCmCMakeBuildTools QUIET CONFIG)

if(NOT ROCmCMakeBuildTools_FOUND)
    message(STATUS "ROCmCMakeBuildTools not found. Fetching from source...")
    include(FetchContent)
    FetchContent_Declare(
        rocm-cmake GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git GIT_TAG rocm-6.4.3
        GIT_SHALLOW TRUE
    )

    FetchContent_GetProperties(rocm-cmake)
    if(NOT rocm-cmake_POPULATED)
        FetchContent_Populate(rocm-cmake)
        message(
            STATUS
                "Added ROCm CMake modules path: ${rocm-cmake_SOURCE_DIR}/share/rocmcmakebuildtools/cmake"
        )
        list(APPEND CMAKE_MODULE_PATH ${rocm-cmake_SOURCE_DIR}/share/rocmcmakebuildtools/cmake)
    endif()
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMCheckTargetIds)
include(ROCMInstallSymlinks)
include(ROCMUtilities)
