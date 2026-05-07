# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################

message( STATUS "Configuring lapack external dependency" )
include( ExternalProject )

# set( lapack_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package )
set( PREFIX_LAPACK ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where lapack should install, defaults to /usr/local" )
set( lapack_cmake_args -DCMAKE_INSTALL_PREFIX=${PREFIX_LAPACK} -DBUILD_SHARED_LIBS=OFF )
append_cmake_cli_arguments( lapack_cmake_args lapack_cmake_args )

set( lapack_git_repository "https://github.com/Reference-LAPACK/lapack-release" CACHE STRING "URL to download lapack from" )
set( lapack_git_tag "lapack-3.9.1" CACHE STRING "git branch" )

# message( STATUS "lapack_make ( " ${lapack_make} " ) " )
# message( STATUS "lapack_cmake_args ( " ${lapack_cmake_args} " ) " )

enable_language( Fortran )
include( GNUInstallDirs )

# lapack cmake exports has a bug on debian architectures, they do not take into account the
# lib/<machine> paths
# if CMAKE_INSTALL_LIBDIR is of the form above, strip the machine
# Match against a '/' in CMAKE_INSTALL_LIBDIR, i.e. lib/x86_64-linux-gnu
if( ${CMAKE_INSTALL_LIBDIR} MATCHES "lib/.*" )
  list( APPEND lapack_cmake_args "-DCMAKE_INSTALL_LIBDIR=lib" )
endif( )

ExternalProject_Add(
  lapack
  PREFIX ${CMAKE_BINARY_DIR}/lapack
  GIT_REPOSITORY ${lapack_git_repository}
  GIT_TAG ${lapack_git_tag}
  CMAKE_ARGS ${lapack_cmake_args} -DCBLAS=ON -DLAPACKE=OFF -DBUILD_TESTING=OFF -DBUILD_INDEX64=ON
  LOG_BUILD 1
  INSTALL_COMMAND ""
  LOG_INSTALL 1
)

ExternalProject_Get_Property( lapack source_dir )

set_property( TARGET lapack PROPERTY FOLDER "extern" )
ExternalProject_Get_Property( lapack install_dir )
ExternalProject_Get_Property( lapack binary_dir )

# For use by the user of ExternalGtest.cmake
set( LAPACK_INSTALL_ROOT ${install_dir} )
set( LAPACK_BINARY_ROOT ${binary_dir} )
