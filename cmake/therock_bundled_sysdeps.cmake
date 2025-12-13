# therock_bundled_sysdeps.cmake
# Configuration for bundled system dependencies.
# When THEROCK_BUNDLE_SYSDEPS is ON, certain system libraries are built and
# bundled with the distribution for portability.
#
# Each bundled library creates a THEROCK_BUNDLED_${name} variable that can be
# included in subproject RUNTIME_DEPS. If bundling is enabled and the platform
# supports it, this will be set to the target name; otherwise it's empty.
#
# For the full list of bundled dependencies, see:
#   docs/development/dependencies.md

# Helper function to declare a bundled system dependency
# Arguments:
#   name - Library name (e.g., BZIP2, ELFUTILS)
#   PLATFORMS - List of supported platforms (e.g., linux, windows)
#
# Creates: THEROCK_BUNDLED_${name} variable
function(therock_declare_bundled_sysdep name)
  cmake_parse_arguments(ARG "" "" "PLATFORMS" ${ARGN})

  # Initialize to empty
  set(THEROCK_BUNDLED_${name} "" PARENT_SCOPE)

  # If bundling is enabled, check if current platform is supported
  if(THEROCK_BUNDLE_SYSDEPS)
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" _platform_lower)

    # Check if current platform is in the supported platforms list
    if(_platform_lower IN_LIST ARG_PLATFORMS)
      # Convert library name to lowercase for target name
      string(TOLOWER "${name}" _lib_lower)
      set(THEROCK_BUNDLED_${name} "therock-${_lib_lower}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

################################################################################
# Platform-Specific Tool Requirements
################################################################################

if(THEROCK_BUNDLE_SYSDEPS)
  message(STATUS "Building with bundled system dependencies enabled")

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # patchelf is required for RPATH manipulation on Linux
    find_program(PATCHELF patchelf)
    if(NOT PATCHELF)
      message(FATAL_ERROR "Building with THEROCK_BUNDLE_SYSDEPS=ON on Linux requires 'patchelf'")
    endif()

    # meson is required for building several system dependencies
    find_program(MESON_BUILD meson)
    if(NOT MESON_BUILD)
      message(FATAL_ERROR "Building with THEROCK_BUNDLE_SYSDEPS=ON on Linux requires 'meson' (easiest: pip install meson)")
    endif()

  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    # Windows-specific build requirements can be added here
    # Currently no special tools required

  else()
    message(FATAL_ERROR "Bundled system deps not supported on this platform (THEROCK_BUNDLE_SYSDEPS=ON)")
  endif()
endif()

################################################################################
# Bundled System Dependency Declarations
################################################################################
# Each declaration creates a THEROCK_BUNDLED_${name} variable for use in
# subproject RUNTIME_DEPS lists.

# Compression libraries
therock_declare_bundled_sysdep(BZIP2 PLATFORMS linux windows)
therock_declare_bundled_sysdep(LIBLZMA PLATFORMS linux)
therock_declare_bundled_sysdep(ZLIB PLATFORMS linux windows)
therock_declare_bundled_sysdep(ZSTD PLATFORMS linux windows)

# System utilities
therock_declare_bundled_sysdep(ELFUTILS PLATFORMS linux)
therock_declare_bundled_sysdep(LIBCAP PLATFORMS linux)
therock_declare_bundled_sysdep(LIBDRM PLATFORMS linux)
therock_declare_bundled_sysdep(NUMACTL PLATFORMS linux)

# Database
therock_declare_bundled_sysdep(SQLITE3 PLATFORMS linux windows)
