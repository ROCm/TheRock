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

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

# ########################################################################
# Main
# ########################################################################
prepend_path( ".." hipblas_headers_public relative_hipblas_headers_public )

if(HIP_PLATFORM STREQUAL amd)
  set( hipblas_source "${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipblas.cpp" )

  message(STATUS "** NOTE: amd_detail/hipblas.cpp is compiled with the verbose flag -v for QC purposes.")
  SET_SOURCE_FILES_PROPERTIES( ${CMAKE_CURRENT_SOURCE_DIR}/amd_detail/hipblas.cpp PROPERTIES COMPILE_FLAGS "-v" )
else( )
  set( hipblas_source "${CMAKE_CURRENT_SOURCE_DIR}/nvidia_detail/hipblas.cpp" )
endif( )

if(BUILD_ADDRESS_SANITIZER)
    add_link_options(-fuse-ld=lld)
endif()

add_library( hipblas
  ${hipblas_source}
  ${CMAKE_CURRENT_SOURCE_DIR}/hipblas_auxiliary.cpp
  ${relative_hipblas_headers_public}
)
add_library( roc::hipblas ALIAS hipblas )

set(static_depends)

find_package( hipblas-common REQUIRED CONFIG PATHS ${ROCM_PATH})
target_link_libraries( hipblas PUBLIC roc::hipblas-common )

# Build hipblas from source on AMD platform
if(HIP_PLATFORM STREQUAL amd)
  if( NOT TARGET rocblas )
    if( CUSTOM_ROCBLAS )
      set ( ENV{rocblas_DIR} ${CUSTOM_ROCBLAS})
      find_package( rocblas REQUIRED CONFIG NO_CMAKE_PATH )
    elseif( WIN32 )
      find_package( rocblas REQUIRED CONFIG PATHS ${ROCBLAS_PATH} ${ROCM_PATH})
    else()
      find_package( rocblas REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm /opt/rocm/rocblas /usr/local/rocblas )
    endif( )
  endif( )

  list(APPEND static_depends PACKAGE rocblas)
  target_link_libraries( hipblas PRIVATE roc::rocblas )
  target_link_libraries( hipblas PUBLIC hip::host )

  # Add rocSOLVER as a dependency if BUILD_WITH_SOLVER is on
  if( BUILD_WITH_SOLVER )
    if( NOT TARGET rocsolver )
      if( CUSTOM_ROCSOLVER)
        set ( ENV{rocsolver_DIR} ${CUSTOM_ROCSOLVER})
        find_package( rocsolver REQUIRED CONFIG NO_CMAKE_PATH )

        # in case of using custom rocsolver and not custom rocblas, we need to have
        # custom rocsolver include directories before rocblas/hip include directories
        # in case there is a rocsolver installed on the system.
        target_include_directories( hipblas
          SYSTEM PRIVATE $<BUILD_INTERFACE:${ROCSOLVER_INCLUDE_DIRS}> )
      elseif(WIN32)
        find_package( rocsolver REQUIRED CONFIG PATHS ${ROCSOLVER_PATH} ${ROCM_PATH})
      else()
        find_package( rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm /opt/rocm/rocsolver /usr/local/rocsolver )
      endif()
    endif( )
    list(APPEND static_depends PACKAGE rocsolver)
    target_link_libraries( hipblas PRIVATE roc::rocsolver )
  endif( )

  if( CUSTOM_TARGET )
    target_link_libraries( hipblas PRIVATE hip::${CUSTOM_TARGET} )
  endif( )

else( )
  target_compile_definitions( hipblas PRIVATE ${HIPBLAS_HIP_PLATFORM_COMPILER_DEFINES} )

  target_link_libraries( hipblas PRIVATE ${CUDA_CUBLAS_LIBRARIES} )

  # External header includes included as system files
  target_include_directories( hipblas
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )
endif( )

# External header includes included as system files
target_include_directories( hipblas
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${ROCBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${ROCSOLVER_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

# Internal header includes
target_include_directories( hipblas
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipblas>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
          $<BUILD_INTERFACE:${HIPBLAS-COMMON_INCLUDE_DIRS}>
  PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}
)

rocm_set_soversion( hipblas ${hipblas_SOVERSION} )
set_target_properties( hipblas PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipblas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

if (WIN32)
  add_custom_command( TARGET hipblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipblas> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipblas> )
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command( TARGET hipblas POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/hipblas.pdb ${PROJECT_BINARY_DIR}/clients/staging/hipblas.pdb )
  endif()
endif()

if(BUILD_CODE_COVERAGE)
  target_compile_options(hipblas PRIVATE -g -O0 -fprofile-instr-generate -fcoverage-mapping)
  target_link_options(hipblas PUBLIC -fprofile-instr-generate)
endif()

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( hipblas PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( hipblas EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipblas/hipblas-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( hipblas PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS hipblas
  INCLUDE
    ${CMAKE_BINARY_DIR}/include
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

if(HIP_PLATFORM STREQUAL amd )
  rocm_export_targets(
    TARGETS roc::hipblas
    DEPENDS PACKAGE hip
    DEPENDS PACKAGE hipblas-common
    STATIC_DEPENDS ${static_depends}
    NAMESPACE roc::
  )
else( )
  rocm_export_targets(
    TARGETS roc::hipblas
    DEPENDS PACKAGE HIP
    DEPENDS PACKAGE hipblas-common
    NAMESPACE roc::
  )
endif( )

# Force installation of .f90 module files
rocm_install(FILES "hipblas_module.f90"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipblas")
