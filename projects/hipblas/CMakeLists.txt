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

# ROCm software requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 17 )

# Build documentation
option(BUILD_DOCS "Build documentation" OFF)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( hipblas LANGUAGES CXX )

find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip ${HIP_DIR}/cmake )

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
include(dependencies)

set ( VERSION_STRING "3.3.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

option( BUILD_VERBOSE "Output additional build information" OFF )

option( BUILD_WITH_SOLVER "Add additional functions from rocSOLVER" ON )

if( BUILD_WITH_SOLVER )
    add_definitions( -D__HIP_PLATFORM_SOLVER__ )
endif( )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build hipBLAS as a shared library" ON )

# Deprecated USE_CUDA option
if(DEFINED USE_CUDA)
  if(USE_CUDA)
    set(ENV{HIP_PLATFORM} nvidia)
    message(DEPRECATED "USE_CUDA is deprecated (use environment variable HIP_PLATFORM=nvidia)")
  else()
    set(ENV{HIP_PLATFORM} amd)
    message(DEPRECATION "USE_CUDA is deprecated (use environment variable HIP_PLATFORM=amd)")
  endif()
endif()

# Hip headers required of all clients; clients use hip to allocate device memory
find_package( hip CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )

# support for cuda backend with hip < 6.0
if(NOT hip_FOUND)
    find_package( HIP MODULE REQUIRED )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
    set(ENV{HIP_PLATFORM} nvidia)
    if (NOT HIP_PLATFORM)
      set(HIP_PLATFORM nvidia)
    endif()
else()
    if (NOT HIP_PLATFORM)
      set(HIP_PLATFORM amd)
    endif()
endif()

if(HIP_PLATFORM STREQUAL nvidia)
  find_package( CUDA REQUIRED )
  set( HIPBLAS_HIP_PLATFORM_COMPILER_DEFINES __HIP_PLATFORM_NVCC__ __HIP_PLATFORM_NVIDIA__ )
endif()

option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)

option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
endif()

if(NOT SKIP_LIBRARY)
  add_subdirectory(library)
endif()

include( clients/cmake/build-options.cmake )

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")

  if(BUILD_FORTRAN_CLIENTS)
    set(GFORTRAN_RPM "libgfortran4")
    set(GFORTRAN_DEB "libgfortran4")
    if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR CLIENTS_OS STREQUAL "almalinux")
      if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
        set(GFORTRAN_RPM "libgfortran")
      endif()
    elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
      set(GFORTRAN_DEB "libgfortran5")
    elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
      set(GFORTRAN_RPM "gfortran")
    endif()
  endif()

  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component(clients-common)
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common)
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common)
  endif()
  if(BUILD_CLIENTS_SAMPLES)
    rocm_package_setup_client_component( 
      samples
      DEPENDS
        COMPONENT clients-common)
  endif()
  if(BUILD_FORTRAN_CLIENTS)
    if(BUILD_CLIENTS_TESTS)
      rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
      rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
    endif()
    if(BUILD_CLIENTS_BENCHMARKS)
      rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
      rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
    endif()
    if(BUILD_CLIENTS_SAMPLES)
      rocm_package_add_rpm_dependencies(COMPONENT samples DEPENDS "${GFORTRAN_RPM}")
      rocm_package_add_deb_dependencies(COMPONENT samples DEPENDS "${GFORTRAN_DEB}")
    endif()
  endif()

  add_subdirectory( clients )
endif( )

# Build docs
if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
    set( CPACK_SOURCE_GENERATOR "ZIP" )
    set( CPACK_GENERATOR "ZIP" )
endif( )

# Package specific CPACK vars
if(HIP_PLATFORM STREQUAL amd)
  set(rocblas_minimum 5.2.0)
  set(rocsolver_minimum 3.32.0)
  rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}" "rocsolver >= ${rocsolver_minimum}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}" "rocsolver-static-devel >= ${rocsolver_minimum}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}" "rocsolver-static-dev >= ${rocsolver_minimum}")
endif( )

set(hipblas_common_minimum 1.4.0)

if(BUILD_SHARED_LIBS)
  rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-dev >= ${hipblas_common_minimum}")
  rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-devel >= ${hipblas_common_minimum}")
else()
  rocm_package_add_deb_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-dev >= ${hipblas_common_minimum}")
  rocm_package_add_rpm_dependencies(COMPONENT devel DEPENDS "hipblas-common-static-devel >= ${hipblas_common_minimum}")
endif()

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT")

if (WIN32)
  if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  endif()
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
else()
  if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
  endif()
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

# Give hipblas compiled for CUDA backend a different name
if(HIP_PLATFORM STREQUAL amd)
    set( package_name hipblas )
else( )
    set( package_name hipblas-alt )
endif( )

set( HIPBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm BLAS marshalling library"
    MAINTAINER "hipBLAS Maintainer <hipblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPBLAS_CONFIG_DIR}
)


#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > make coverage GTEST_FILTER=<>
  # will run:
  #  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
  #  > make coverage_output (generate html documentation)
  #

  #
  # Run coverage analysis
  #
  set(coverage_test ./clients/staging/hipblas-test)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(coverage_test ./clients/staging/hipblas-test-d)
  endif()

  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage/profraw
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/hipblas-coverage_%m.profraw" ${coverage_test} --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis hipblas)

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  add_custom_target(
    coverage
    DEPENDS coverage_analysis
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/hipblas-coverage_*.profraw -o ./coverage-report/hipblas.profdata
    COMMAND ${LLVM_COV} report -object ./library/src/libhipblas.so -instr-profile=./coverage-report/hipblas.profdata
    COMMAND ${LLVM_COV} show -object ./library/src/libhipblas.so -instr-profile=./coverage-report/hipblas.profdata -format=html -output-dir=coverage-report
    COMMAND ${LLVM_COV} export -object ./library/src/libhipblas.so -instr-profile=./coverage-report/hipblas.profdata -format=lcov > ./coverage-report/coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
