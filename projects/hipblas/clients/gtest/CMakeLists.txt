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

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

set(hipblas_test_source
  hipblas_gtest_main.cpp
  hipblas_test.cpp
  auxil/auxiliary_gtest.cpp
  auxil/set_get_mode_gtest.cpp
  auxil/set_get_matrix_vector_gtest.cpp
  blas1/asum_gtest.cpp
  blas1/axpy_gtest.cpp
  blas1/copy_gtest.cpp
  blas1/dot_gtest.cpp
  blas1/iamaxmin_gtest.cpp
  blas1/nrm2_gtest.cpp
  blas1/rot_gtest.cpp
  blas1/scal_gtest.cpp
  blas1/swap_gtest.cpp
  blas2/gbmv_gtest.cpp
  blas2/gemv_gtest.cpp
  blas2/ger_gtest.cpp
  blas2/hbmv_gtest.cpp
  blas2/hemv_gtest.cpp
  blas2/her_gtest.cpp
  blas2/her2_gtest.cpp
  blas2/hpmv_gtest.cpp
  blas2/hpr_gtest.cpp
  blas2/hpr2_gtest.cpp
  blas2/sbmv_gtest.cpp
  blas2/spmv_gtest.cpp
  blas2/spr_gtest.cpp
  blas2/spr2_gtest.cpp
  blas2/symv_gtest.cpp
  blas2/syr_gtest.cpp
  blas2/syr2_gtest.cpp
  blas2/tbmv_gtest.cpp
  blas2/tbsv_gtest.cpp
  blas2/tpmv_gtest.cpp
  blas2/tpsv_gtest.cpp
  blas2/trmv_gtest.cpp
  blas2/trsv_gtest.cpp
  blas3/dgmm_gtest.cpp
  blas3/gemm_gtest.cpp
  blas3/hemm_gtest.cpp
  blas3/geam_gtest.cpp
  blas3/herk_gtest.cpp
  blas3/her2k_gtest.cpp
  blas3/herkx_gtest.cpp
  blas3/symm_gtest.cpp
  blas3/syrk_gtest.cpp
  blas3/syr2k_gtest.cpp
  blas3/syrkx_gtest.cpp
  blas3/trsm_gtest.cpp
  blas3/trmm_gtest.cpp
  blas3/trtri_gtest.cpp
  blas_ex/axpy_ex_gtest.cpp
  blas_ex/dot_ex_gtest.cpp
  blas_ex/nrm2_ex_gtest.cpp
  blas_ex/rot_ex_gtest.cpp
  blas_ex/scal_ex_gtest.cpp
  blas_ex/trsm_ex_gtest.cpp
  blas_ex/gemm_ex_gtest.cpp
  blas_ex/syrk_ex_gtest.cpp
  blas_ex/herk_ex_gtest.cpp
)

if( BUILD_WITH_SOLVER )
  set( hipblas_solver_test_source
    solver/getrf_gtest.cpp
    solver/getrs_gtest.cpp
    solver/getri_gtest.cpp
    solver/geqrf_gtest.cpp
    solver/gels_gtest.cpp
  )
endif( )

set( hipblas_test_common
  ../common/utility.cpp
  ../common/cblas_interface.cpp
  ../common/norm.cpp
  ../common/unit.cpp
  ../common/near.cpp
  ../common/arg_check.cpp
  ../common/argument_model.cpp
  ../common/hipblas_arguments.cpp
  ../common/hipblas_parse_data.cpp
  ../common/hipblas_datatype2string.cpp
  ../common/host_alloc.cpp
  ${BLIS_CPP}
)

if(BUILD_FORTRAN_CLIENTS)
  add_executable( hipblas-test ${hipblas_test_source} ${hipblas_solver_test_source} ${hipblas_test_common} $<TARGET_OBJECTS:hipblas_fortran_client> )
else()
  add_executable( hipblas-test ${hipblas_test_source} ${hipblas_solver_test_source} ${hipblas_test_common} )
endif()

target_include_directories( hipblas-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_compile_definitions( hipblas-test PRIVATE GOOGLE_TEST ${BLIS_DEFINES} )

# External header includes included as SYSTEM files
target_include_directories( hipblas-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${FLAME_INCLUDE_DIR}>
    ${ROCM_PATH}/include
)

target_link_libraries( hipblas-test PRIVATE roc::hipblas GTest::gtest GTest::gtest_main )

if (BUILD_FORTRAN_CLIENTS)
    target_link_libraries( hipblas-test PRIVATE hipblas_fortran_client )
endif()

# need mf16c flag for float->half convertion
target_compile_options( hipblas-test PRIVATE -mf16c ) # -Wno-deprecated-declarations )

target_compile_options(hipblas-test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)

target_compile_definitions( hipblas-test PRIVATE ${COMMON_DEFINES} )

target_link_libraries( hipblas-test PRIVATE ${BLAS_LIBRARY} ${COMMON_LINK_LIBS} )
apply_omp_settings( hipblas-test )

if (NOT WIN32)
    target_link_libraries( hipblas-test PRIVATE stdc++fs )
endif()

if(HIP_PLATFORM STREQUAL amd)
  target_link_libraries( hipblas-test PRIVATE hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipblas-test PRIVATE hip::${CUSTOM_TARGET} )
  endif( )

else( )
  target_include_directories( hipblas-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipblas-test PRIVATE ${CUDA_LIBRARIES} )
endif( )

if (CREATE_TEST_APP_LOCAL_DEPLOY AND WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${LAPACK_DIR}/bin/*.dll
    ${BLIS_DIR}/lib/*.dll
    ${OPENBLAS_DIR}/bin/*.dll
    ${HIP_DIR}/bin/amd*.dll
    ${HIP_DIR}/bin/hiprt*.dll
    ${HIP_DIR}/bin/hipinfo.exe
    ${ROCBLAS_PATH}/bin/rocblas*.dll
    ${ROCSOLVER_PATH}/bin/rocsolver*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET hipblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )

  if(DEFINED ROCBLAS_PATH)
    add_custom_command( TARGET hipblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_directory ${ROCBLAS_PATH}/bin/rocblas/library/ ${PROJECT_BINARY_DIR}/staging/library/)
  else()
    message(WARNING "ROCBLAS_PATH not set. rocBLAS Tensile runtime kernels will not be copied to client staging directory (insitu testing affected).")
  endif()
endif()

set_target_properties( hipblas-test PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARED_REQUIRED ON
    CXX_EXTENSIONS OFF
    IMPORT_PREFIX ""
    IMPORT_SUFFIX ".lib"
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set( HIPBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/staging/hipblas_gtest.data")

set( HIPBLAS_AUX_YAML_DATA auxil/set_get_matrix_vector_gtest.yaml auxil/set_get_mode_gtest.yaml )

set( HIPBLAS_L1_YAML_DATA blas1/asum_gtest.yaml blas1/axpy_gtest.yaml blas1/copy_gtest.yaml
                          blas1/dot_gtest.yaml  blas1/iamaxmin_gtest.yaml blas1/nrm2_gtest.yaml
                          blas1/rot_gtest.yaml  blas1/scal_gtest.yaml blas1/swap_gtest.yaml )
set( HIPBLAS_L2_YAML_DATA blas2/gbmv_gtest.yaml blas2/gemv_gtest.yaml blas2/ger_gtest.yaml
                          blas2/hbmv_gtest.yaml blas2/hemv_gtest.yaml blas2/her_gtest.yaml
                          blas2/her2_gtest.yaml blas2/hpmv_gtest.yaml blas2/hpr_gtest.yaml
                          blas2/hpr2_gtest.yaml blas2/sbmv_gtest.yaml blas2/spmv_gtest.yaml
                          blas2/spr_gtest.yaml  blas2/spr2_gtest.yaml blas2/symv_gtest.yaml
                          blas2/syr_gtest.yaml  blas2/syr2_gtest.yaml blas2/tbmv_gtest.yaml
                          blas2/tbsv_gtest.yaml blas2/tpmv_gtest.yaml blas2/tpsv_gtest.yaml
                          blas2/trmv_gtest.yaml blas2/trsv_gtest.yaml )

set( HIPBLAS_L3_YAML_DATA blas3/dgmm_gtest.yaml blas3/geam_gtest.yaml blas3/gemm_gtest.yaml
                          blas3/hemm_gtest.yaml blas3/herk_gtest.yaml blas3/her2k_gtest.yaml
                          blas3/herkx_gtest.yaml blas3/symm_gtest.yaml blas3/syrk_gtest.yaml
                          blas3/syr2k_gtest.yaml blas3/syrkx_gtest.yaml blas3/trmm_gtest.yaml
                          blas3/trsm_gtest.yaml blas3/trtri_gtest.yaml )

set( HIPBLAS_EX_YAML_DATA blas_ex/axpy_ex_gtest.yaml blas_ex/dot_ex_gtest.yaml blas_ex/nrm2_ex_gtest.yaml
                          blas_ex/rot_ex_gtest.yaml blas_ex/scal_ex_gtest.yaml blas_ex/gemm_ex_gtest.yaml 
                          blas_ex/syrk_ex_gtest.yaml blas_ex/herk_ex_gtest.yaml blas_ex/trsm_ex_gtest.yaml )

if( BUILD_WITH_SOLVER )
  set( HIPBLAS_SOLVER_YAML_DATA solver/gels_gtest.yaml solver/geqrf_gtest.yaml solver/getrf_gtest.yaml solver/getri_gtest.yaml solver/getrs_gtest.yaml )
endif()

add_custom_command( OUTPUT "${HIPBLAS_TEST_DATA}"
                    COMMAND ${Python3_EXECUTABLE} ../common/hipblas_gentest.py -I ../include hipblas_gtest.yaml -o "${HIPBLAS_TEST_DATA}"
                    DEPENDS ../common/hipblas_gentest.py ../include/hipblas_common.yaml "${HIPBLAS_AUX_YAML_DATA}" "${HIPBLAS_L1_YAML_DATA}" "${HIPBLAS_L2_YAML_DATA}" "${HIPBLAS_L3_YAML_DATA}" "${HIPBLAS_EX_YAML_DATA}" "${HIPBLAS_SOLVER_YAML_DATA}" hipblas_gtest.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

add_custom_target( hipblas-test-data DEPENDS "${HIPBLAS_TEST_DATA}" )

add_dependencies( hipblas-test hipblas-test-data hipblas-clients-common )

rocm_install(TARGETS hipblas-test COMPONENT tests)
rocm_install(FILES ${HIPBLAS_TEST_DATA} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT tests)
