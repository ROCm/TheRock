# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ROCm Emulation Tools

if(THEROCK_ENABLE_ROCJITSU)
  ##############################################################################
  # rocjitsu
  #
  # ROCm Just-in-Time Suite: simulation, virtualization, and instrumentation
  # tools for AMD GPU applications. Provides ISA-level execution of AMD GPU
  # and RISC-V machine code from HIP fat binaries without requiring GPU
  # hardware.
  ##############################################################################

  therock_cmake_subproject_declare(rocjitsu
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/emulation/rocjitsu"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocjitsu"
    CMAKE_ARGS
      -DBUILD_TESTING=${THEROCK_BUILD_TESTING}
    RUNTIME_DEPS
      ${THEROCK_BUNDLED_LIBDRM}
  )

  therock_cmake_subproject_glob_c_sources(rocjitsu
    SUBDIRS
      lib
      tools
      tests
  )

  therock_cmake_subproject_activate(rocjitsu)

  therock_cmake_subproject_build_test(rocjitsu
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --no-tests=error
  )

  therock_test_validate_shared_lib(PATH rocjitsu/dist/lib
    LIB_NAMES
      librocjitsu.so
      librocjitsu_hooks.so
  )

  therock_provide_artifact(rocjitsu
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rocjitsu.toml
    COMPONENTS
      dbg
      dev
      lib
      run
    SUBPROJECT_DEPS
      rocjitsu
  )
endif()

if(THEROCK_ENABLE_MIRAGE)
  ##############################################################################
  # mirage
  #
  # User-facing UX (CLI + embedded web dashboard) for the rocjitsu GPU emulator
  # and other emulator backends. The project is a Rust workspace wrapped by a
  # thin CMake layer that drives cargo; the build produces the unified `mirage`
  # binary. Building the dashboard SPA requires Node.js 20.19+ and npm on PATH.
  ##############################################################################

  therock_cmake_subproject_declare(mirage
    DISABLE_AMDGPU_TARGETS
    NO_MERGE_COMPILE_COMMANDS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/emulation/mirage"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/mirage"
    CMAKE_ARGS
      -DMIRAGE_CARGO_PROFILE=release
  )

  therock_cmake_subproject_activate(mirage)

  therock_cmake_subproject_build_test(mirage
    COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure --no-tests=error
  )

  therock_provide_artifact(mirage
    TARGET_NEUTRAL
    DESCRIPTOR artifact-mirage.toml
    COMPONENTS
      dbg
      dev
      run
    SUBPROJECT_DEPS
      mirage
  )
endif()
