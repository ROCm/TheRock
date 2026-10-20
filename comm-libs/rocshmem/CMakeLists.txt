# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rocSHMEM is declared in its own directory, apart from the rest of comm-libs,
# because it must be added before profiler/ while rccl must be added after it
# (rccl depends on the profiler). See the root CMakeLists.txt for the ordering.
# add_subdirectory() is one-shot per directory, so the two cannot share a file.

if(THEROCK_ENABLE_ROCSHMEM)
  ##############################################################################
  # rocSHMEM
  ##############################################################################
  set(_rocshmem_subproject_names)
  # amdsmi is optional: only available on Linux and when explicitly enabled.
  set(_rocshmem_optional_runtime_deps)
  if(NOT WIN32)
    if(TARGET amdsmi)
      list(APPEND _rocshmem_optional_runtime_deps amdsmi)
    endif()
  endif()
  therock_cmake_subproject_declare(rocshmem
    # rocshmem uses .hip_fatbin/.hip_gpubin_* ELF sections for device code
    # instead of .hipFatBinSegment, making it incompatible with kpack injection.
    # Build all GPU targets in a single target-neutral artifact as a workaround.
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocshmem"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}"
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
      # Register functional/unit tests with CTest and install the relocatable
      # bin/rocshmem/**/CTestTestfile.cmake files (with quick/standard/
      # comprehensive/full tier labels) that test_runner.py drives via ctest.
      -DBUILD_CTESTS=${THEROCK_BUILD_TESTING}
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
    RUNTIME_DEPS
      hip-clr
      # rocshmem registers its API tables with rocprofiler-register
      # (USE_ROCPROFILER_REGISTER, ON by default) so rocprofiler-sdk can trace
      # its host-stream APIs. The find_package for it is QUIET, so without this
      # dep the build succeeds with tracing silently disabled.
      rocprofiler-register
      ${THEROCK_BUNDLED_NUMACTL}
      ${_rocshmem_optional_runtime_deps}
  )
  therock_cmake_subproject_glob_c_sources(rocshmem
    SUBDIRS
      src
  )
  # rocSHMEM installs lib/cmake/rocshmem/rocshmem-config.cmake, and every
  # consumer (rocprofiler-sdk among them) does find_package(rocshmem). The
  # dependency provider matches the advertised name verbatim, so it must be
  # spelled the same way here or those lookups escape to the system resolver.
  therock_cmake_subproject_provide_package(rocshmem rocshmem lib/cmake/rocshmem)
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
