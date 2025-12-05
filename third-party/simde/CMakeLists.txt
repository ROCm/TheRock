# Dual-mode CMakeLists.txt for simde (header-only library)
#
# NOTE: simde upstream uses meson, but we bypass it entirely because:
# 1. simde is header-only - no compilation needed
# 2. meson adds complexity and cross-platform issues (Windows link.exe errors, etc.)
# 3. All we need is to copy headers and generate a simple pkg-config file
# 4. Manual installation is simpler, faster, and identical across all platforms
#
# This approach is uniform on Linux and Windows - no surprises.

# IMPORTANT: Keep version in sync with URL below
set(SIMDE_VERSION "0.8.2")

# Section 1: Super-project integration (when included from TheRock)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Fetch the simde source tarball
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-simde-sources
    SOURCE_DIR "${_source_dir}"
    # Originally from: https://github.com/simd-everywhere/simde/archive/refs/tags/v${SIMDE_VERSION}.tar.gz
    # NOTE: Version must match SIMDE_VERSION variable above
    URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/simde-${SIMDE_VERSION}.tar.gz
    URL_HASH SHA256=ed2a3268658f2f2a9b5367628a85ccd4cf9516460ed8604eed369653d49b25fb
    TOUCH "${_download_stamp}"
  )

  # Declare the sub-project (uses this same CMakeLists.txt as sub-build)
  therock_cmake_subproject_declare(therock-simde
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR .
    BINARY_DIR build
    NO_MERGE_COMPILE_COMMANDS
    BACKGROUND_BUILD
    OUTPUT_ON_FAILURE
    CMAKE_ARGS
      "-DSOURCE_DIR=${_source_dir}"
      "-DSIMDE_VERSION=${SIMDE_VERSION}"
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
project(SIMDE_BUILD VERSION ${SIMDE_VERSION})

# Manual installation for header-only library
# No meson, no compilation - just copy headers and generate pkg-config

# Generate pkg-config file directly
file(CONFIGURE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/simde.pc"
  CONTENT "prefix=\${pcfiledir}/../..
includedir=\${prefix}/include

Name: SIMDe
Description: Portable SIMD wrapper library
Version: @PROJECT_VERSION@
Cflags: -I\${includedir}
"
  @ONLY
)

# Install headers
install(
  DIRECTORY "${SOURCE_DIR}/simde"
  DESTINATION include
)

# Install pkg-config file
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/simde.pc"
  DESTINATION lib/pkgconfig
)
