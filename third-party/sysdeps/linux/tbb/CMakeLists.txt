# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Super-project block: download sources and declare the sub-project.
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-tbb-sources
    SOURCE_DIR "${_source_dir}"
    # Originally mirrored from: https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2022.3.0.tar.gz
    URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/oneTBB-2022.3.0.tar.gz
    URL_HASH SHA256=01598a46c1162c27253a0de0236f520fd8ee8166e9ebb84a4243574f88e6e50a
    TOUCH "${_download_stamp}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  therock_cmake_subproject_declare(therock-tbb
    USE_DIST_AMDGPU_TARGETS
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    EXTERNAL_SOURCE_DIR .
    FPRINT_SOURCE_DIR "${_source_dir}"
    FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
    BINARY_DIR build
    CMAKE_ARGS
      "-DSOURCE_DIR=${_source_dir}"
      "-DPATCHELF=${PATCHELF}"
      "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
    INSTALL_DESTINATION
      lib/rocm_sysdeps
    INTERFACE_LINK_DIRS
      lib/rocm_sysdeps/lib
    INTERFACE_INSTALL_RPATH_DIRS
      lib/rocm_sysdeps/lib
    INTERFACE_PKG_CONFIG_DIRS
      lib/rocm_sysdeps/lib/pkgconfig
    EXTRA_DEPENDS
      "${_download_stamp}"
  )

  therock_cmake_subproject_provide_package(therock-tbb TBB lib/rocm_sysdeps/lib/cmake/TBB)
  therock_cmake_subproject_activate(therock-tbb)

  therock_test_validate_shared_lib(
    PATH build/dist/lib/rocm_sysdeps/lib
    LIB_NAMES librocm_sysdeps_tbb.so
  )

  return()
endif()

# Sub-project block: invoked by TheRock's cmake sub-project machinery.
cmake_minimum_required(VERSION 3.25)
project(TBB_BUILD)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

add_custom_target(build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  # Clean any previous install and create a fresh working copy of the source.
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  # Patch source: rename library output names to rocm_sysdeps_ prefix.
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s"
  # Configure oneTBB.
  COMMAND
    "${CMAKE_COMMAND}"
      -S "${CMAKE_CURRENT_BINARY_DIR}/s"
      -B "${CMAKE_CURRENT_BINARY_DIR}/tbb-build"
      -G Ninja
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DTBB_TEST=OFF"
  # Build.
  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/tbb-build" --verbose
  # Install.
  COMMAND
    "${CMAKE_COMMAND}" --install "${CMAKE_CURRENT_BINARY_DIR}/tbb-build"
  # Patch install: fix RPATH, relocate .pc, remove upstream cmake config.
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}" --
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" "${CMAKE_INSTALL_PREFIX}"
)
