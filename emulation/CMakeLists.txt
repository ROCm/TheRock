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
      librocjitsu_kmd.so
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
