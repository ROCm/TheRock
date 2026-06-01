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

function( get_lapack lapack_lib lapack_inc )
  if(EXISTS  "${BUILD_DIR}/deps/deps-install/lib/liblapack64.a")
    set( lib "${BUILD_DIR}/deps/deps-install/lib/liblapack64.a" )
    set( inc "${BUILD_DIR}/deps/deps-install/include" )
    set( ${lapack_inc} ${inc} PARENT_SCOPE )
  else()
    set(BLA_SIZEOF_INTEGER 8)
    find_package( LAPACK REQUIRED )
    set( lib "${LAPACK_LIBRARIES}" )
  endif()
  set( ${lapack_lib} ${lib} PARENT_SCOPE )
  enable_language( Fortran )
endfunction( )

function( get_cblas cblas_libs cblas_inc cblas_defs )
  if(LINK_BLIS)
    if(EXISTS    "${BUILD_DIR}/deps/blis/lib/libblis.a")
        set( libs ${BUILD_DIR}/deps/blis/lib/libblis.a )
        set( inc  ${BUILD_DIR}/deps/blis/include/blis )
        set( ${cblas_inc} ${inc} PARENT_SCOPE )
      elseif(EXISTS "/usr/local/lib/libblis.a")
        set( libs    /usr/local/lib/libblis.a )
        set( inc     /usr/local/include/blis )
        set( ${cblas_inc} ${inc} PARENT_SCOPE )
    endif()
  else()
    if(EXISTS "${BUILD_DIR}/deps/deps-install/lib/libcblas64.a" AND EXISTS "${BUILD_DIR}/deps/deps-install/lib/libblas64.a")
      set( libs ${BUILD_DIR}/deps/deps-install/lib/libcblas64.a ${BUILD_DIR}/deps/deps-install/lib/libblas64.a )
      set( inc "${BUILD_DIR}/deps/deps-install/include" )
      set( ${cblas_inc} ${inc} PARENT_SCOPE )
    else()
      # get BLAS
      set(BLA_SIZEOF_INTEGER 8)
      find_package( BLAS REQUIRED )
      set( libs ${BLAS_LIBRARIES} )

      # get CBLAS from the same source as BLAS
      get_filename_component(BLAS_DIR "${BLAS_LIBRARIES}" DIRECTORY)
      set(ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
      if(BLA_STATIC)
        if(WIN32)
          list(PREPEND CMAKE_FIND_LIBRARY_SUFFIXES .lib .a)
        else()
          set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
        endif()
      endif()
      find_library(CBLAS_LIBRARIES NAMES cblas64 PATHS BLAS_DIR)
      set(CMAKE_FIND_LIBRARY_SUFFIXES ${ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})

      if(CBLAS_LIBRARIES)
        list( PREPEND libs ${CBLAS_LIBRARIES} )
      endif()
    endif()

    # defines to set cblas.h to 64-bit indices
    set ( defs "WeirdNEC" "LAPACK_ILP64" "USE64BITINT" "OPENBLAS_USE64BITINT")
  endif()
  set( ${cblas_libs} ${libs} PARENT_SCOPE )
  set( ${cblas_defs} ${defs} PARENT_SCOPE )
endfunction( )

function( apply_omp_settings lib_target_ )
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::OpenMP_CXX)
    set_target_properties( ${lib_target_} PROPERTIES
      BUILD_RPATH "${HIP_CLANG_ROOT}/lib"
    )
    set_target_properties( ${lib_target_} PROPERTIES
      INSTALL_RPATH "$ORIGIN/../llvm/lib"
    )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::omp)
    set_target_properties( ${lib_target_} PROPERTIES
      BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}"
    )
    set_target_properties( ${lib_target_} PROPERTIES
      INSTALL_RPATH "$ORIGIN/../llvm/${openmp_LIB_DIR}"
    )
  endif()
endfunction()


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

# This project may compile dependencies for clients
if ( BUILD_FORTRAN_CLIENTS )
  if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
    set( CMAKE_Fortran_COMPILER  "gfortran" )
  endif()
  set( fortran_language "Fortran" )
endif()
project( hipblas-clients LANGUAGES CXX ${fortran_language} )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 17 )

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

if( BUILD_FORTRAN_CLIENTS )
    set(hipblas_f90_source_clients_no_solver
      include/hipblas_fortran_no_solver_module.f90
    )

    set(hipblas_f90_source_clients_solver
      include/hipblas_fortran_module.f90
    )

    set (hipblas_f90_source
      ../library/src/hipblas_module.f90
    )
endif()

if( BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_SAMPLES )
  # Create hipBLAS Fortran module
  if( BUILD_FORTRAN_CLIENTS )
      # Set Fortran module output directory
      set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/hipblas)
      add_library(hipblas_fortran OBJECT ${hipblas_f90_source})
  endif()

  if( BUILD_FORTRAN_CLIENTS )
    if( BUILD_WITH_SOLVER )
      add_library(hipblas_fortran_client STATIC ${hipblas_f90_source_clients_solver} $<TARGET_OBJECTS:hipblas_fortran>)
    else()
      add_library(hipblas_fortran_client STATIC ${hipblas_f90_source_clients_no_solver} $<TARGET_OBJECTS:hipblas_fortran>)
    endif()
  endif()

  if(SKIP_LIBRARY)
    include_directories(${HIPBLAS_LIBRARY_DIR}/include/hipblas)
  else()
    include_directories(${CMAKE_BINARY_DIR}/include/hipblas)
    include_directories(${CMAKE_BINARY_DIR}/include)
  endif()
endif( )

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)

  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Look for openmp config in ROCm install to populate openmp_LIB_DIR and openmp_LIB_INSTALL_DIR
    find_package(OpenMP CONFIG PATHS "${HIP_CLANG_ROOT}/lib/cmake")
  endif()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::omp)
    set( COMMON_LINK_LIBS "OpenMP::omp")
    message(STATUS "Found openmp-config.cmake at ${OpenMP_DIR}")
  else()
    # if it fails to find OpenMP compile and link flags in strange configurations it can just use non-parallel reference computation
    # if there is no omp.h to find the client compilation will fail and this should be obvious, used to be REQUIRED
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
      set( COMMON_LINK_LIBS "OpenMP::OpenMP_CXX")
    endif()
  endif()

  if (TARGET Threads::Threads)
    list( APPEND COMMON_LINK_LIBS "Threads::Threads")
  endif()

  set( COMMON_DEFINES HIPBLAS_USE_HIP_HALF HIPBLAS_BFLOAT16_CLASS ROCM_USE_FLOAT16 HIPBLAS_NO_DEPRECATED_WARNINGS ${HIPBLAS_HIP_PLATFORM_COMPILER_DEFINES} )
  if (BUILD_FORTRAN_CLIENTS)
    list( APPEND COMMON_DEFINES "HIPBLAS_FORTRAN_CLIENTS" )
  endif()

  # Default search for static host libraries
  if(NOT DEFINED BLA_STATIC)
    set(BLA_STATIC ON)
  endif()

  if( NOT WIN32 )
    if (LINK_BLIS)
      if(EXISTS          "${BUILD_DIR}/deps/amd-blis/lib/ILP64/libflame.a" AND EXISTS "${BUILD_DIR}/deps/amd-blis/lib/ILP64/libblis-mt.a") # 4.0 and 4.1.0
        set( BLAS_LIBRARY ${BUILD_DIR}/deps/amd-blis/lib/ILP64/libflame.a ${BUILD_DIR}/deps/amd-blis/lib/ILP64/libblis-mt.a )
        set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/amd-blis/include/ILP64 )
        list( APPEND COMMON_DEFINES "FLA_ENABLE_ILP64")
      elseif(EXISTS "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libflame.a" AND EXISTS "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a" AND EXISTS "/opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libaoclutils.a" )
        set( BLAS_LIBRARY -Wl,--allow-multiple-definition /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libflame.a /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libblis-mt.a /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/lib_ILP64/libaoclutils.a )
        set( BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-gcc-4.2.0/gcc/include_ILP64/ )
        list( APPEND COMMON_DEFINES "FLA_ENABLE_ILP64")
      elseif(EXISTS      "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libflame.a" AND EXISTS "/opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a" )
        set( BLAS_LIBRARY /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libflame.a              /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/lib_ILP64/libblis-mt.a )
        set( BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.1.0/aocc/include_ILP64/ )
        list( APPEND COMMON_DEFINES "FLA_ENABLE_ILP64")
      elseif(EXISTS      "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libflame.a" AND EXISTS "/opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a" )
        set( BLAS_LIBRARY /opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libflame.a              /opt/AMD/aocl/aocl-linux-aocc-4.0/lib_ILP64/libblis-mt.a )
        set( BLIS_INCLUDE_DIR /opt/AMD/aocl/aocl-linux-aocc-4.0/include_ILP64/ )
        list( APPEND COMMON_DEFINES "FLA_ENABLE_ILP64")
      else()
        # fallbacks include earlier blis
        get_lapack( LAPACK_LIB LAPACK_INCLUDE_DIR )
        get_cblas( BLAS_LIBRARY BLAS_INCLUDE_DIR BLAS_DEFINES )
        list( PREPEND BLAS_LIBRARY ${LAPACK_LIB} )
        list( APPEND COMMON_DEFINES ${BLAS_DEFINES} )
      endif()
    else()
      # Linking lapack library requires fortran flags
      get_lapack( LAPACK_LIB LAPACK_INCLUDE_DIR )
      get_cblas( BLAS_LIBRARY BLAS_INCLUDE_DIR BLAS_DEFINES )
      list( PREPEND BLAS_LIBRARY ${LAPACK_LIB} )
      list( APPEND COMMON_DEFINES ${BLAS_DEFINES} )
    endif()
  else() # WIN32
    file(TO_CMAKE_PATH "C:/Program\ Files/AMD/AOCL-Windows" AOCL_ROOT)
    if (LINK_BLIS AND EXISTS "C:/Program\ Files/AMD/AOCL-Windows/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT.lib" AND EXISTS "C:/Program\ Files/AMD/AOCL-Windows/amd-libflame/lib/ILP64/AOCL-LibFlame-Win-MT.lib" AND EXISTS "C:/Program\ Files/AMD/AOCL-Windows/amd-utils/lib/libaoclutils_static.lib" )
      set( BLAS_LIBRARY "-l\"C:/Program\ Files/AMD/AOCL-Windows/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT\"" "-l\"C:/Program\ Files/AMD/AOCL-Windows/amd-libflame/lib/ILP64/AOCL-LibFlame-Win-MT\"" "-l\"C:/Program\ Files/AMD/AOCL-Windows/amd-utils/lib/libaoclutils_static\"" )
      set( BLIS_INCLUDE_DIR "C:/Program\ Files/AMD/AOCL-Windows/amd-blis/include/ILP64" )
      set( FLAME_INCLUDE_DIR "C:/Program\ Files/AMD/AOCL-Windows/amd-libflame/include/ILP64" )
      set( BLIS_DEFINES BLIS_ENABLE_NO_UNDERSCORE_API BLIS_ENABLE_CBLAS )
    else()
      set( BLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include CACHE PATH "OpenBLAS library include path" )
      find_library( BLAS_LIBRARY libopenblas
                    PATHS ${OPENBLAS_DIR}/lib
                    NO_DEFAULT_PATH
                  )
      if (NOT BLAS_LIBRARY)
        find_package( OpenBLAS CONFIG REQUIRED )
        set( BLAS_LIBRARY OpenBLAS::OpenBLAS )
        set( BLAS_INCLUDE_DIR "" )
      endif()
    endif()
  endif()

  message(STATUS "CLIENT COMMON_DEFINES: ${COMMON_DEFINES}")
  message(STATUS "CLIENT COMMON CXX_OPTIONS: ${COMMON_CXX_OPTIONS}")
  message(STATUS "CLIENT COMMON LINK: ${COMMON_LINK_LIBS}")

  if ( DEFINED BLIS_INCLUDE_DIR )
    set( BLIS_CPP ../common/blis_interface.cpp )
  endif()

  if(EXISTS  "${BUILD_DIR}/deps/deps-install/lib/libgtest.a")
    set( GTEST_ROOT "${BUILD_DIR}/deps/deps-install")
  endif()
  find_package( GTest REQUIRED )

  message(STATUS "Build Dir: ${BUILD_DIR}")
  message(STATUS "LINK_BLIS: ${LINK_BLIS}")
  message(STATUS "Linking Libs: ${BLAS_LIBRARY}")

  if( NOT TARGET hipblas )
    find_package( hipblas REQUIRED CONFIG PATHS ${HIPBLAS_LIBRARY_DIR} )
  endif( )

  if( BUILD_CLIENTS_TESTS )
    add_subdirectory( gtest )
  endif( )

  if( BUILD_CLIENTS_BENCHMARKS )
    add_subdirectory( benchmarks )
  endif( )

endif()

set( HIPBLAS_COMMON "${PROJECT_BINARY_DIR}/staging/hipblas_common.yaml")
add_custom_command( OUTPUT "${HIPBLAS_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipblas_common.yaml "${HIPBLAS_COMMON}"
                    DEPENDS include/hipblas_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPBLAS_TEMPLATE "${PROJECT_BINARY_DIR}/staging/hipblas_template.yaml")
add_custom_command( OUTPUT "${HIPBLAS_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipblas_template.yaml "${HIPBLAS_TEMPLATE}"
                    DEPENDS include/hipblas_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPBLAS_SMOKE "${PROJECT_BINARY_DIR}/staging/hipblas_smoke.yaml")
add_custom_command( OUTPUT "${HIPBLAS_SMOKE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipblas_smoke.yaml "${HIPBLAS_SMOKE}"
                    DEPENDS include/hipblas_smoke.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPBLAS_GENTEST "${PROJECT_BINARY_DIR}/staging/hipblas_gentest.py")
add_custom_command( OUTPUT "${HIPBLAS_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/hipblas_gentest.py "${HIPBLAS_GENTEST}"
                    DEPENDS common/hipblas_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( HIPBLAS_CLIENTS_README "${PROJECT_BINARY_DIR}/staging/hipblas_clients_readme.txt")
add_custom_command( OUTPUT "${HIPBLAS_CLIENTS_README}"
                    COMMAND ${CMAKE_COMMAND} -E copy hipblas_clients_readme.txt "${HIPBLAS_CLIENTS_README}"
                    DEPENDS hipblas_clients_readme.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )


add_custom_target( hipblas-clients-common DEPENDS "${HIPBLAS_COMMON}" "${HIPBLAS_TEMPLATE}" "${HIPBLAS_SMOKE}" "${HIPBLAS_GENTEST}" "${HIPBLAS_CLIENTS_README}" )

if( BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  rocm_install(
    FILES ${HIPBLAS_COMMON} ${HIPBLAS_TEMPLATE} ${HIPBLAS_SMOKE}
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients-common
  )
  rocm_install(
    PROGRAMS ${HIPBLAS_GENTEST}
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients-common
  )
  # this readme also serves to prevent an empty package hipblas-clients which dpkg may auto-remove entire hipblas-clients and non empty children
  rocm_install(
    FILES ${HIPBLAS_CLIENTS_README}
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT clients
  )
endif()
