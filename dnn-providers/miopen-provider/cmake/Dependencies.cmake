# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

# Only setup dependencies for standalone builds
if(NOT BUILD_PLUGIN_AS_DEPENDENCY AND NOT HIPDNN_SKIP_TESTS)
    include(FetchContent)

    # Try to find GTest first
    find_package(GTest CONFIG QUIET)

    if(NOT GTest_FOUND)
        message(STATUS "Fetching GTest for standalone plugin build")

        fetchcontent_declare(
            googletest URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
                           DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )

        set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
        set(INSTALL_GTEST OFF CACHE INTERNAL "")
        set(BUILD_GMOCK ON CACHE INTERNAL "")

        fetchcontent_makeavailable(googletest)
    else()
        message(STATUS "Found system GTest")
    endif()
endif()
