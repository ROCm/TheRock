# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_ROCDECODE AND THEROCK_ENABLE_SYSDEPS_AMD_MESA)
  ##############################################################################
  # rocdecode library
  ##############################################################################

  # rocprofiler-register is not built on Windows (see base/CMakeLists.txt), and
  # rocDecode disables its rocprofiler-register integration there.
  set(_rocdecode_rocprofiler_register_dep rocprofiler-register)
  if(WIN32)
    set(_rocdecode_rocprofiler_register_dep)
  endif()

  therock_cmake_subproject_declare(rocdecode
    FPRINT_PROJECT_SOURCE
    FPRINT_PROJECT_OPTIONAL_INPUTS ${THEROCK_PROJECT_ROOT_INPUTS}
    FPRINT_PROJECT_ENVIRONMENT
      ${THEROCK_PROJECT_ROCM_VERSION_ENVIRONMENT} "AMDGPU_TARGETS" "ASAN_OPTIONS"
      "ASAN_SYMBOLIZER_PATH" "FFMPEG_ROOT" "GPU_ARCHS" "HIP_DEVICE_LIB_PATH" "LSAN_OPTIONS" "PATH"
      "PKG_CONFIG_PATH" "ProgramFiles" "ROCM_PATH" "ROCPROFILER_REGISTER_ROOT" "USERPROFILE"
      "rocprofiler_register_ROOT"
    USE_DIST_AMDGPU_TARGETS
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocdecode"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocdecode"
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      ${_rocdecode_rocprofiler_register_dep}
      ${THEROCK_BUNDLED_LIBDRM}
      ${THEROCK_BUNDLED_AMDMESA}
      hip-clr
    BUILD_DEPS
    CMAKE_ARGS
      -DROCM_PATH=
      -DBUILD_TESTS=ON
    INTERFACE_LINK_DIRS
      "lib"
    INSTALL_RPATH_DIRS
      "lib;lib/rocm_sysdeps/lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib;lib/rocm_sysdeps/lib"
  )
  therock_cmake_subproject_glob_c_sources(rocdecode
    SUBDIRS
    .
  )
  therock_cmake_subproject_provide_package(rocdecode rocdecode lib/cmake/rocdecode)
  therock_cmake_subproject_activate(rocdecode)
  therock_test_validate_shared_lib(
    PATH ${CMAKE_CURRENT_BINARY_DIR}/rocdecode/dist/lib
    LIB_NAMES
      librocdecode.so
  )

  therock_provide_artifact(rocdecode
    TARGET_NEUTRAL
    DESCRIPTOR
      artifact-rocdecode.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      test
    SUBPROJECT_DEPS
      ${THEROCK_BUNDLED_AMDMESA}
      rocdecode
  )
endif(THEROCK_ENABLE_ROCDECODE AND THEROCK_ENABLE_SYSDEPS_AMD_MESA)

if(THEROCK_ENABLE_ROCJPEG AND THEROCK_ENABLE_SYSDEPS_AMD_MESA)
  ##############################################################################
  # rocjpeg library
  ##############################################################################

  therock_cmake_subproject_declare(rocjpeg
    FPRINT_PROJECT_SOURCE
    FPRINT_PROJECT_OPTIONAL_INPUTS ${THEROCK_PROJECT_ROOT_INPUTS}
    FPRINT_PROJECT_ENVIRONMENT
      ${THEROCK_PROJECT_ROCM_VERSION_ENVIRONMENT} "AMDGPU_TARGETS" "GPU_ARCHS" "HIP_DEVICE_LIB_PATH"
      "ROCM_PATH" "ROCPROFILER_REGISTER_ROOT" "rocprofiler_register_ROOT"
    USE_DIST_AMDGPU_TARGETS
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocjpeg"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocjpeg"
    COMPILER_TOOLCHAIN
      amd-hip
    RUNTIME_DEPS
      rocprofiler-register
      ${THEROCK_BUNDLED_LIBDRM}
      ${THEROCK_BUNDLED_AMDMESA}
      hip-clr
    BUILD_DEPS
    CMAKE_ARGS
      -DROCM_PATH=
      -DBUILD_TESTS=ON
    INTERFACE_LINK_DIRS
      "lib"
    INSTALL_RPATH_DIRS
      "lib;lib/rocm_sysdeps/lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib;lib/rocm_sysdeps/lib"
  )
  therock_cmake_subproject_glob_c_sources(rocjpeg
    SUBDIRS
    .
  )
  therock_cmake_subproject_provide_package(rocjpeg rocjpeg lib/cmake/rocjpeg)
  therock_cmake_subproject_activate(rocjpeg)
  therock_test_validate_shared_lib(
    PATH ${CMAKE_CURRENT_BINARY_DIR}/rocjpeg/dist/lib
    LIB_NAMES
      librocjpeg.so
  )
  therock_provide_artifact(rocjpeg
    TARGET_NEUTRAL
    DESCRIPTOR
      artifact-rocjpeg.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      test
    SUBPROJECT_DEPS
      ${THEROCK_BUNDLED_AMDMESA}
      rocjpeg
  )
endif(THEROCK_ENABLE_ROCJPEG AND THEROCK_ENABLE_SYSDEPS_AMD_MESA)
