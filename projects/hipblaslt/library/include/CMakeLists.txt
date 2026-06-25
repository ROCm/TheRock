# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt/hipblaslt-version.h"
)

generate_export_header(hipblaslt
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/hipblaslt/hipblaslt-export.h"
    BASE_NAME hipblaslt
)

target_sources(hipblaslt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/hipblaslt/hipblaslt-export.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt_float8.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt-ext.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt-ext-op.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt-types.h>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/hipblaslt/hipblaslt-version.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt-export.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt-ext.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt-ext-op.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt_float8.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt/hipblaslt-types.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblaslt-version.h>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt/hipblaslt_xfloat32.h"
)
