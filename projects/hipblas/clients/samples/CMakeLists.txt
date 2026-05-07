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

set( hipblas_samples_common ../common/utility.cpp )

add_executable( hipblas-example-sscal example_sscal.cpp ${hipblas_samples_common} )
add_executable( hipblas-example-scal-ex example_scal_ex.cpp ${hipblas_samples_common} )
add_executable( hipblas-example-sgemm example_sgemm.cpp ${hipblas_samples_common} )
add_executable( hipblas-example-strmm example_strmm.cpp ${hipblas_samples_common} )
add_executable( hipblas-example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${hipblas_samples_common} )
add_executable( hipblas-example-c example_c.c ${hipblas_samples_common} )
add_executable( hipblas-example-hip-complex-her2 example_hip_complex_her2.cpp ${hipblas_samples_common} )
add_executable( hipblas-example-hgemm-half example_hgemm_hip_half.cpp ${hipblas_samples_common})
add_executable( hipblas-example-gemmEx example_gemm_ex.cpp ${hipblas_samples_common})

if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/amdclang\\+\\+$")
  add_executable( hipblas-example-hgemm example_hgemm.cpp ${hipblas_samples_common} )
endif( )

if(NOT WIN32)
  add_executable( hipblas-example-sscal-fortran example_sscal_fortran.F90 $<TARGET_OBJECTS:hipblas_fortran_client>)
  add_executable( hipblas-example-gemmEx-fortran example_gemm_ex_fortran.F90 $<TARGET_OBJECTS:hipblas_fortran_client>)

  set( sample_list_fortran hipblas-example-sscal-fortran hipblas-example-gemmEx-fortran )
endif()

if(HIP_PLATFORM STREQUAL amd)
  add_executable(hipblas-example-bfdot-hip-bfloat16 example_bfdot_hip_bfloat16.cpp ${hipblas_samples_common})
  endif()

# We test for C99 compatibility in the example_c.c test
set_source_files_properties(example_c.c PROPERTIES LANGUAGE CXX)
set_source_files_properties(example_c.c PROPERTIES COMPILE_FLAGS "-xc -std=c99")
# needed for hipcc to find hip includes for C example
target_compile_options(hipblas-example-c PRIVATE -I${HIP_INCLUDE_DIRS})

# We test for C++11 compatibility in one of the samples
#set_source_files_properties(example_sgemm_strided_batched.cpp PROPERTIES COMPILE_FLAGS "-std=c++11")

if( NOT TARGET hipblas )
  find_package( hipblas CONFIG PATHS ${HIPBLAS_LIBRARY_DIR} )
  if( NOT hipblas_FOUND )
    message( FATAL_ERROR "hipBLAS is a required dependency and is not found; try adding rocblas path to CMAKE_PREFIX_PATH")
  endif( )
endif( )

list (APPEND hipblas-example-executables hipblas-example-sscal hipblas-example-scal-ex hipblas-example-strmm hipblas-example-sgemm hipblas-example-sgemm-strided-batched hipblas-example-gemmEx hipblas-example-hip-complex-her2 hipblas-example-hgemm-half hipblas-example-c ${sample_list_fortran} )
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/amdclang\\+\\+$")
  list (APPEND hipblas-example-executables hipblas-example-hgemm)
endif( )
if(HIP_PLATFORM STREQUAL amd)
  list (APPEND hipblas-example-executables hipblas-example-bfdot-hip-bfloat16)
  target_compile_options(hipblas-example-bfdot-hip-bfloat16 PRIVATE -DHIPBLAS_USE_HIP_BFLOAT16)
endif()

target_compile_options( hipblas-example-hgemm-half PRIVATE -DHIPBLAS_USE_HIP_HALF )
target_compile_options( hipblas-example-gemmEx PRIVATE -DHIPBLAS_USE_HIP_HALF )

# HIP on Windows: xhip is required with clang++ to get __half defined
if ( WIN32 )
target_compile_options( hipblas-example-hgemm-half PRIVATE -xhip )
target_compile_options( hipblas-example-gemmEx PRIVATE -xhip )
endif()

foreach( exe IN LISTS hipblas-example-executables)

  # External header includes included as SYSTEM files
  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )

  target_include_directories( ${exe} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> )

  target_compile_options( ${exe} PRIVATE -mf16c -DHIPBLAS_BFLOAT16_CLASS )
  target_compile_definitions( ${exe} PRIVATE HIPBLAS_NO_DEPRECATED_WARNINGS )

  target_link_libraries( ${exe} PRIVATE roc::hipblas )

  if(HIP_PLATFORM STREQUAL amd)
    target_link_libraries( ${exe} PRIVATE hip::host )

    if( CUSTOM_TARGET )
      target_link_libraries( ${exe} PRIVATE hip::${CUSTOM_TARGET} )
    endif( )

  else( )
    target_compile_definitions( ${exe} PRIVATE ${HIPBLAS_HIP_PLATFORM_COMPILER_DEFINES} )

    target_include_directories( ${exe}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )

    target_link_libraries( ${exe} PRIVATE ${CUDA_LIBRARIES} )
  endif( )

  set_target_properties( ${exe} PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS ON
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

  rocm_install(TARGETS ${exe} COMPONENT samples)

endforeach( )
