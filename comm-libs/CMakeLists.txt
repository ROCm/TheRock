# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_RCCL)
  # Most libraries need to depend on the profiler, but is conditionally only used
  # on Posix.
  set(optional_profiler_deps)

  if(THEROCK_FLAG_INCLUDE_PROFILER)
    list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
  endif()

  # RCCL can optionally use amdsmi/rocm_smi_lib on Linux, but those may not be
  # enabled/declared depending on feature selection.
  set(_rccl_optional_runtime_deps)
  if(NOT WIN32)
    if(TARGET amdsmi)
      list(APPEND _rccl_optional_runtime_deps amdsmi)
    endif()
    if(TARGET rocm_smi_lib)
      list(APPEND _rccl_optional_runtime_deps rocm_smi_lib)
    endif()
  endif()

  ##############################################################################
  # rccl
  ##############################################################################
  set(_rccl_subproject_names)

  therock_cmake_subproject_declare(rccl
    # Build all GPU targets in a single target-neutral artifact as long as the
    # host code isn't fully target agnostic.
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rccl"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rccl"
    # High latency LTO link of a single library.
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
      # TODO(#56): Enable once reworked.
      -DENABLE_MSCCLPP=OFF
      -DENABLE_MSCCL_KERNEL=OFF
      -DBUILD_TESTS=${THEROCK_BUILD_TESTING}
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-fmt
      therock-googletest
    RUNTIME_DEPS
      hip-clr
      hipify
      rocprofiler-register
      ${_rccl_optional_runtime_deps}
      ${optional_profiler_deps}
  )
  therock_cmake_subproject_glob_c_sources(rccl
    SUBDIRS
      src
  )
  therock_cmake_subproject_provide_package(rccl RCCL lib/cmake/rccl)
  therock_cmake_subproject_activate(rccl)
  list(APPEND _rccl_subproject_names rccl)

  if(THEROCK_BUILD_TESTING)

    set(_rccl_tests_optional_deps)
    if(THEROCK_ENABLE_MPI)
      set(_rccl_tests_optional_deps "therock-openmpi")
    endif()

    therock_cmake_subproject_declare(rccl-tests
      USE_TEST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rccl-tests"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rccl-tests"
      BACKGROUND_BUILD
      CMAKE_ARGS
        -DHIP_PLATFORM=amd
        -DROCM_PATH=
#        -DROCM_DIR=
        "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
        -DUSE_MPI=${THEROCK_ENABLE_MPI}
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        rocm-cmake
      RUNTIME_DEPS
        hip-clr
        hipify
        rccl
        ${_rccl_tests_optional_deps}
    )
    therock_cmake_subproject_glob_c_sources(rccl-tests
      SUBDIRS
        src
    )
    therock_cmake_subproject_activate(rccl-tests)
    list(APPEND _rccl_subproject_names rccl-tests)
  endif(THEROCK_BUILD_TESTING)

  therock_provide_artifact(rccl
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rccl.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      ${_rccl_subproject_names}
  )
endif()

if(THEROCK_ENABLE_ROCSHMEM)
  ##############################################################################
  # rocSHMEM
  ##############################################################################
  set(_rocshmem_subproject_names)
  therock_cmake_subproject_declare(rocshmem
    # rocshmem uses .hip_fatbin/.hip_gpubin_* ELF sections for device code
    # instead of .hipFatBinSegment, making it incompatible with kpack injection.
    # Build all GPU targets in a single target-neutral artifact as a workaround.
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocshmem"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocshmem"
    # High latency LTO link of a single library.
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DCMAKE_VERBOSE_MAKEFILE=OFF
      -DUSE_RO=ON
      -DUSE_IPC=ON
      -DUSE_GDA=ON
      -DGDA_MLX5=ON
      -DGDA_BNXT=ON
      -DGDA_IONIC=ON
      # When using MPI (e.g. for bootsrapping or using RO) it only works with Open MPI and is
      # incompatible with MPICH, Intel MPI, or MVAPICH.
      # The loading mechanisms for using MPI is using dlopen() at run-time. It only support
      # Open MPI 5.x.x and 4.1.x symbols.
      # There is no compile time dependency to MPI.
      -DUSE_EXTERNAL_MPI=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
      -DBUILD_FUNCTIONAL_TESTS=${THEROCK_BUILD_TESTING}
      -DBUILD_UNIT_TESTS=${THEROCK_BUILD_TESTING}
      -DSOURCE_DIR=${_source_dir}
      -DPATCHELF=${PATCHELF}
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-fmt
      therock-googletest
      ${THEROCK_BUNDLED_NUMACTL}
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(rocshmem
    SUBDIRS
      src
  )
  therock_cmake_subproject_provide_package(rocshmem ROCSHMEM lib/cmake/rocshmem)
  therock_cmake_subproject_activate(rocshmem)
  list(APPEND _rocshmem_subproject_names rocshmem)

  therock_provide_artifact(rocshmem
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rocshmem.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      ${_rocshmem_subproject_names}
  )
endif(THEROCK_ENABLE_ROCSHMEM)
