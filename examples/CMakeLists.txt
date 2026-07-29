# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TODO: HIP language support not based on an /opt/rocm installed SDK
# must have these three cache variables set to avoid language setup
# error. https://github.com/ROCm/TheRock/issues/102
set(HIP_CMAKE_ARGS
  -DCMAKE_HIP_PLATFORM=amd
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${THEROCK_BINARY_DIR}/dist/rocm"
)

if(WIN32)
  # On Windows, only a unified compiler is supported (i.e. use the HIP compiler
  # for everything).
  list(APPEND HIP_CMAKE_ARGS
    "-DCMAKE_CXX_COMPILER=${THEROCK_BINARY_DIR}/dist/rocm/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  )
else()
  # On everything else, the HIP language compiler can be independent from the
  # overall CXX compiler.
  list(APPEND HIP_CMAKE_ARGS
    "-DCMAKE_HIP_COMPILER=${THEROCK_BINARY_DIR}/dist/rocm/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  )
endif()

# Per-arch libraries disabled for the selected targets (e.g. the portable
# amdgcnspirv target) are not built, so the example must not find_package() them.
function(_therock_example_available out_var enabled project_name)
  therock_project_disabled(_skip "${project_name}")
  if(_skip)
    set("${out_var}" OFF PARENT_SCOPE)
  else()
    set("${out_var}" "${enabled}" PARENT_SCOPE)
  endif()
endfunction()

_therock_example_available(_blas_available "${THEROCK_ENABLE_BLAS}" rocBLAS)
_therock_example_available(_fft_available "${THEROCK_ENABLE_FFT}" rocFFT)
_therock_example_available(_hipdnn_available "${THEROCK_ENABLE_HIPDNN}" hipDNN)
_therock_example_available(_miopen_available "${THEROCK_ENABLE_MIOPEN}" MIOpen)
_therock_example_available(_prim_available "${THEROCK_ENABLE_PRIM}" rocPRIM)
_therock_example_available(_rand_available "${THEROCK_ENABLE_RAND}" rocRAND)
_therock_example_available(_rccl_available "${THEROCK_ENABLE_RCCL}" rccl)
_therock_example_available(_rocwmma_available "${THEROCK_ENABLE_ROCWMMA}" rocWMMA)
_therock_example_available(_rocalution_available "${THEROCK_ENABLE_ROCALUTION}" rocALUTION)
_therock_example_available(_libhipcxx_available "${THEROCK_ENABLE_LIBHIPCXX}" libhipcxx)
_therock_example_available(_solver_available "${THEROCK_ENABLE_SOLVER}" rocSOLVER)
_therock_example_available(_sparse_available "${THEROCK_ENABLE_SPARSE}" rocSPARSE)

# hipTensor is additionally gated on composable_kernel (a hard build dependency);
# _ck_enabled is set by math-libs/CMakeLists.txt.
set(_hiptensor_available OFF)
if(_ck_enabled AND THEROCK_ENABLE_HIPTENSOR)
  _therock_example_available(_hiptensor_available ON hipTensor)
endif()

# Examples are configured against the installed ROCm prefix. The
# cpp-sdk-user project requires HIP (LANGUAGES CXX HIP), so skip when
# HIP is not enabled (e.g. in early multi-arch CI stages) or when
# sanitizers are enabled (sanitizer builds do not install HIP).
if(THEROCK_ENABLE_HIP_RUNTIME AND "${THEROCK_SANITIZER}" STREQUAL "")
  add_test(
    NAME therock-examples-cpp-sdk-user
    COMMAND
      "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH=${THEROCK_BINARY_DIR}/dist/rocm" --
      "${CMAKE_COMMAND}"
          "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cpp-sdk-user"
          "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cpp-sdk-user"
          "-DGENERATOR=${CMAKE_GENERATOR}"
          "-DTHEROCK_ENABLE_BLAS=${_blas_available}"
          "-DTHEROCK_ENABLE_FFT=${_fft_available}"
          "-DTHEROCK_ENABLE_HIP=${THEROCK_ENABLE_HIP_RUNTIME}"
          "-DTHEROCK_ENABLE_HIPDNN=${_hipdnn_available}"
          "-DTHEROCK_ENABLE_MIOPEN=${_miopen_available}"
          "-DTHEROCK_ENABLE_HIPTENSOR=${_hiptensor_available}"
          "-DTHEROCK_ENABLE_PRIM=${_prim_available}"
          "-DTHEROCK_ENABLE_RAND=${_rand_available}"
          "-DTHEROCK_ENABLE_RCCL=${_rccl_available}"
          "-DTHEROCK_ENABLE_ROCWMMA=${_rocwmma_available}"
          "-DTHEROCK_ENABLE_ROCALUTION=${_rocalution_available}"
          "-DTHEROCK_ENABLE_LIBHIPCXX=${_libhipcxx_available}"
          "-DTHEROCK_ENABLE_SOLVER=${_solver_available}"
          "-DTHEROCK_ENABLE_SPARSE=${_sparse_available}"
          ${HIP_CMAKE_ARGS}
        -P "${CMAKE_CURRENT_SOURCE_DIR}/clean_configure_test_project.cmake"
  )
endif()
