# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_library(getopt::getopt INTERFACE IMPORTED GLOBAL)

if(WIN32)
    include(FetchContent)

    FetchContent_Declare(
            getopt
            GIT_REPOSITORY https://github.com/apwojcik/getopt.git
            GIT_TAG main
            SYSTEM
        )

    set(__build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

    FetchContent_MakeAvailable(getopt)

    # Restore the old value of BUILD_SHARED_LIBS
    set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)

    FetchContent_GetProperties(getopt)

    target_link_libraries(getopt::getopt INTERFACE wingetopt)
    target_include_directories(getopt::getopt INTERFACE ${getopt_SOURCE_DIR}/src)
endif()