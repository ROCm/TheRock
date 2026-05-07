# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.11)
project(hipblas-common LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(dependencies)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(VERSION_STRING "1.4.0")
rocm_setup_version(VERSION ${VERSION_STRING})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix" FORCE)
    endif()
endif()

add_library(hipblas-common INTERFACE)
add_library(roc::hipblas-common ALIAS hipblas-common)

target_sources(
    hipblas-common
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library/include/hipblas-common/hipblas-common.h>
)
target_include_directories(
    hipblas-common INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/library/include>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

rocm_install_targets(TARGETS hipblas-common INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/library/include)

rocm_export_targets(TARGETS roc::hipblas-common NAMESPACE roc::)

if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SET_DESTDIR FALSE)
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
elseif(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")
set(HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "Path placed into ldconfig file"
)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
    NAME hipblas-common
    DESCRIPTION "Common files for hipBLAS libraries"
    MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
    HEADER_ONLY
)
