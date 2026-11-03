# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Fortran bindings for the ROCm libraries (RFC0015). Each binding is a
# standalone CMake project that lives next to the library it wraps and builds
# against that library once installed. They are built here, after those
# libraries, rather than inside their builds.

if(THEROCK_ENABLE_RUNTIME_FORTRAN)
  ##############################################################################
  # runtime-fortran
  # The HIP and ROCTx bindings (rocm-systems projects/fortran), built against
  # hip-clr and rocprofiler-sdk. Skipped until projects/fortran exists in the
  # rocm-systems source; the artifact is then empty.
  ##############################################################################
  set(_runtime_fortran_subproject_names)
  if(EXISTS "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/fortran/CMakeLists.txt")
    therock_cmake_subproject_declare(runtime-fortran
      USE_TEST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/fortran"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/runtime-fortran"
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DBUILD_FORTRAN_BINDINGS=ON
        -DBUILD_FORTRAN_CLIENTS=${THEROCK_BUILD_TESTING}
        -DCMAKE_PREFIX_PATH="${THEROCK_BINARY_DIR}/dist/rocm"
        -DROCM_PATH="${THEROCK_BINARY_DIR}/dist/rocm"
      COMPILER_TOOLCHAIN
        amd-hip
      RUNTIME_DEPS
        hip-clr
        rocprofiler-sdk
    )
    therock_cmake_subproject_provide_package(runtime-fortran hip-fortran lib/cmake/hip-fortran)
    therock_cmake_subproject_provide_package(runtime-fortran roctx-fortran lib/cmake/roctx-fortran)
    therock_cmake_subproject_activate(runtime-fortran)
    list(APPEND _runtime_fortran_subproject_names runtime-fortran)
  endif()

  therock_provide_artifact(runtime-fortran
    TARGET_NEUTRAL
    DESCRIPTOR artifact-runtime-fortran.toml
    COMPONENTS
      dev
    SUBPROJECT_DEPS
      ${_runtime_fortran_subproject_names}
  )
endif(THEROCK_ENABLE_RUNTIME_FORTRAN)
