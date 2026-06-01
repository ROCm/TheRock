# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

message( STATUS "Configuring gtest external dependency" )
include( ExternalProject )

# set( gtest_cmake_args -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package )
set( PREFIX_GTEST ${CMAKE_INSTALL_PREFIX} CACHE PATH "Location where boost should install, defaults to /usr/local" )
set( gtest_cmake_args -DCMAKE_INSTALL_PREFIX=${PREFIX_GTEST} -DBUILD_SHARED_LIBS=OFF )
append_cmake_cli_arguments( gtest_cmake_args gtest_cmake_args )

set( gtest_git_repository "https://github.com/google/googletest.git" CACHE STRING "URL to download gtest from" )
set( gtest_git_tag "release-1.11.0" CACHE STRING "URL to download gtest from" )

if( MSVC )
  list( APPEND gtest_cmake_args -Dgtest_force_shared_crt=ON )
endif( )

if( CMAKE_CONFIGURATION_TYPES )
  set( gtest_make
        COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
        COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Debug
  )
else( )
  # Add build thread in addition to the number of cores that we have
  include( ProcessorCount )
  ProcessorCount( Cores )

  # If we are not using an IDE, assume nmake with visual studio
  if( MSVC )
    set( gtest_make "nmake" )
  else( )
    set( gtest_make "make" )

    # The -j paramter does not work with nmake
    if( NOT Cores EQUAL 0 )
      math( EXPR Cores "${Cores} + 1 " )
      list( APPEND gtest_make -j ${Cores} )
    else( )
      # If we could not detect # of cores, assume 1 core and add an additional build thread
      list( APPEND gtest_make -j 2 )
    endif( )
  endif( )

  message( STATUS "ExternalGmock using ( " ${Cores} " ) cores to build with" )
endif( )

# message( STATUS "gtest_make ( " ${gtest_make} " ) " )
# message( STATUS "gtest_cmake_args ( " ${gtest_cmake_args} " ) " )

# Master branch has a new structure that combines googletest with googlemock
ExternalProject_Add(
  googletest
  PREFIX ${CMAKE_BINARY_DIR}/gtest
  GIT_REPOSITORY ${gtest_git_repository}
  GIT_TAG ${gtest_git_tag}
  CMAKE_ARGS ${gtest_cmake_args}
  BUILD_COMMAND ${gtest_make}
  LOG_BUILD 1
  INSTALL_COMMAND ""
  LOG_INSTALL 1
)

ExternalProject_Get_Property( googletest source_dir )

# For visual studio, the path 'debug' is hardcoded because that is the default VS configuration for a build.
# Doesn't matter if its the gtest or gtestd project above
set( package_dir "${PREFIX_GTEST}" )
if( CMAKE_CONFIGURATION_TYPES )
  # Create a package by bundling libraries and header files
  if( BUILD_64 )
    set( LIB_DIR lib64 )
  else( )
    set( LIB_DIR lib )
  endif( )

  set( gtest_lib_dir "<BINARY_DIR>/${LIB_DIR}" )
  ExternalProject_Add_Step( googletest createPackage
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Debug ${package_dir}/${LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Release ${package_dir}/${LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Debug ${package_dir}/${LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${gtest_lib_dir}/Release ${package_dir}/${LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${package_dir}/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/gtest/include/gtest ${package_dir}/include/gtest
    DEPENDEES install
  )
endif( )

set_property( TARGET googletest PROPERTY FOLDER "extern")
ExternalProject_Get_Property( googletest install_dir )
ExternalProject_Get_Property( googletest binary_dir )

# For use by the user of ExternalGtest.cmake
set( GTEST_INSTALL_ROOT ${install_dir} )
set( GTEST_BINARY_ROOT ${binary_dir} )
