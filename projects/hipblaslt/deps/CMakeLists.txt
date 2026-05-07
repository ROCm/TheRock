# ########################################################################
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Helper cmake script to automate building dependencies for rocblaslt
# This script can be invoked manually by the user with 'cmake -P'

cmake_minimum_required(VERSION 3.22...3.25.2 FATAL_ERROR)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake )

# Consider removing this in the future
# It can be annoying for visual studio developers to build a project that tries to install into 'program files'
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# The superbuild does not build anything itself; all compiling is done in external projects
project( rocblaslt-dependencies NONE )

option( BUILD_GTEST "Download and build googletest library" ON )
option( BUILD_LAPACK "Download and build lapack library" ON )

# This module scrapes the CMakeCache.txt file and attempts to get all the cli options the user specified to cmake invocation
include( get-cli-arguments )

# The following is a series of super-build projects; this cmake project will download and build
if( BUILD_GTEST )
  include( external-gtest )

  list( APPEND hipblaslt_dependencies googletest )
  set( gtest_custom_target COMMAND cd ${GTEST_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

if( BUILD_LAPACK )
  include( external-lapack )

  list( APPEND hipblaslt_dependencies lapack )
  set( lapack_custom_target COMMAND cd ${LAPACK_BINARY_ROOT}$<SEMICOLON> ${CMAKE_COMMAND} --build . --target install )
endif( )

# POLICY CMP0037 - "Target names should not be reserved and should match a validity pattern"
# Familiar target names like 'install' should be OK at the super-build level
if( POLICY CMP0037 )
  cmake_policy( SET CMP0037 OLD )
endif( )

add_custom_target( install
  ${gtest_custom_target}
  ${lapack_custom_target}
  DEPENDS ${rocblaslt_dependencies}
)
