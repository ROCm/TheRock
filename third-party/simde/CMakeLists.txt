# Dual-mode CMakeLists.txt for simde (meson-based)
# Section 1: Super-project integration (when included from TheRock)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Fetch the simde source tarball
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-simde-sources
    SOURCE_DIR "${_source_dir}"
    # TODO: Mirror to https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/simde-0.8.2.tar.gz
    # Originally from: https://github.com/simd-everywhere/simde/archive/refs/tags/v0.8.2.tar.gz
    URL https://github.com/simd-everywhere/simde/archive/refs/tags/v0.8.2.tar.gz
    URL_HASH SHA256=ed2a3268658f2f2a9b5367628a85ccd4cf9516460ed8604eed369653d49b25fb
    TOUCH "${_download_stamp}"
  )

  # Declare the sub-project (uses this same CMakeLists.txt as sub-build)
  therock_cmake_subproject_declare(therock-simde
    EXTERNAL_SOURCE_DIR .
    BINARY_DIR build
    NO_MERGE_COMPILE_COMMANDS
    BACKGROUND_BUILD
    OUTPUT_ON_FAILURE
    CMAKE_ARGS
      "-DSOURCE_DIR=${_source_dir}"
      "-DMESON_BUILD=${MESON_BUILD}"
      "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
    INTERFACE_PKG_CONFIG_DIRS
      lib/pkgconfig
    EXTRA_DEPENDS
      "${_download_stamp}"
  )
  therock_cmake_subproject_activate(therock-simde)

  add_dependencies(therock-third-party therock-simde)
  return()
endif()

# Section 2: Sub-project build (invoked by super-project)
cmake_minimum_required(VERSION 3.25)
project(SIMDE_BUILD)

if(NOT MESON_BUILD)
  message(FATAL_ERROR "Missing MESON_BUILD from super-project")
endif()

# Meson refuses to build if source dir is subdir of build dir
set(PATCH_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/../patch_source")

add_custom_target(
  meson_build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    # Clean previous build
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # Copy sources (in case we need to patch in the future)
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # Meson setup - simde is header-only, so just need to install headers
    "${CMAKE_COMMAND}" -E chdir "${PATCH_SOURCE_DIR}"
    "${MESON_BUILD}" setup "${CMAKE_CURRENT_BINARY_DIR}"
      --reconfigure
      # We generate relocatable, arch neutral directory layouts
      --prefix "/"
      -Dpkgconfig.relocatable=true
      -Dlibdir=lib
      -Dtests=false
  COMMAND
    # Build (should be minimal for header-only)
    "${MESON_BUILD}" compile --verbose
  COMMAND
    # Install headers using DESTDIR
    "${CMAKE_COMMAND}" -E env "DESTDIR=${CMAKE_INSTALL_PREFIX}" --
      "${MESON_BUILD}" install
)
