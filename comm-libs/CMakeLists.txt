# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Note: rocSHMEM is declared in comm-libs/rocshmem, which the top-level
# CMakeLists.txt adds before `profiler` (so rocprofiler-sdk can optionally trace
# it). It is therefore already declared by the time rccl references it below.

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

  # rccl performs an optional integration with rocSHMEM (ENABLE_ROCSHMEM /
  # ENABLE_ROCSHMEM_GIN) for its GIN/GDA transport. Declare rocshmem as a
  # dependency when available so it resolves from the super-project dist rather
  # than escaping to a stray build tree (rccl's ROCSHMEM.cmake otherwise falls
  # back to a git worktree / ExternalProject checkout of a pinned rocSHMEM
  # commit). rocshmem is declared in comm-libs/rocshmem (added before this
  # directory) so the target exists here.
  set(_rccl_optional_rocshmem_deps)
  if(THEROCK_ENABLE_ROCSHMEM)
    list(APPEND _rccl_optional_rocshmem_deps rocshmem)
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
      ${_rccl_optional_rocshmem_deps}
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
