# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # setup submodule project
    set(_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../amd-mesa/mesa-fork")

    therock_cmake_subproject_declare(therock-amd-mesa
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DMESON_BUILD=${MESON_BUILD}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_INCLUDE_DIRS
        lib/rocm_sysdeps/include
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
    )

    therock_cmake_subproject_activate(therock-amd-mesa)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES rocm_sysdeps_va.dll rocm_sysdeps_va_win32.dll
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(MESA_BUILD)
if(NOT MESON_BUILD)
  message(FATAL_ERROR "Missing MESON_BUILD from super-project")
endif()

# Locate a pkg-config executable that Meson will accept. Meson rejects
# Strawberry Perl's pkg-config.BAT wrapper, so we need a native .exe.
# Preference order:
#   1. PKG_CONFIG env var (set by the user to a known-good executable)
#   2. pkg-config.exe on PATH (e.g. installed via winget/choco/scoop)
if(DEFINED ENV{PKG_CONFIG} AND NOT "$ENV{PKG_CONFIG}" MATCHES "\\.BAT$|\\.bat$")
  set(_meson_pkg_config "PKG_CONFIG=$ENV{PKG_CONFIG}")
  message(STATUS "Using pkg-config from PKG_CONFIG env: $ENV{PKG_CONFIG}")
else()
  find_program(PKG_CONFIG_EXECUTABLE
    NAMES pkg-config.exe pkg-config
    NO_CMAKE_FIND_ROOT_PATH
  )
  if(PKG_CONFIG_EXECUTABLE AND NOT PKG_CONFIG_EXECUTABLE MATCHES "\\.BAT$|\\.bat$")
    set(_meson_pkg_config "PKG_CONFIG=${PKG_CONFIG_EXECUTABLE}")
    message(STATUS "Found pkg-config for mesa: ${PKG_CONFIG_EXECUTABLE}")
  else()
    message(WARNING "No suitable pkg-config found. Meson may fail to resolve libva-win32. "
                    "Set PKG_CONFIG env var to a pkg-config.exe path.")
    set(_meson_pkg_config "")
  endif()
endif()

include("${THEROCK_SOURCE_DIR}/cmake/therock_meson_env.cmake")
therock_get_meson_compiler_env(_meson_cc _meson_cxx)

set(PATCH_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/../patch_source")

add_custom_target(
  meson_build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # We have to patch the sources so make a fresh copy.
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 1: Initial meson setup to download libva-2.22.0 subproject.
    # Required before patching because patch_source.ps1 modifies files
    # in the libva subproject after extraction.
    "${CMAKE_COMMAND}" -E env
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      ${_meson_pkg_config}
      --
      "${MESON_BUILD}" setup
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        -Dva-libs-path="/lib"
        -Dallow-fallback-for=libva
        -Dforce_fallback_for=libva
        -Degl=disabled
        -Dgallium-drivers=d3d12
        -Dgallium-va=enabled
        -Dllvm=disabled
        -Dopengl=false
        -Dvideo-codecs=all
        -Dvulkan-drivers=
        -Dlibva:with_glx=no
        -Dlibva:with_wayland=no
        -Dlibva:with_x11=no
        -Dspirv-tools=disabled
        -Dzlib=disabled
        -Dzstd=disabled
        -Dexpat=disabled
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 2: Patch source files including libva win32 header fix.
    powershell -ExecutionPolicy Bypass -File
      "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.ps1"
      "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 3: Reconfigure meson to build from patched source.
    "${CMAKE_COMMAND}" -E env
      "CC=${_meson_cc}"
      "CXX=${_meson_cxx}"
      ${_meson_pkg_config}
      --
      "${MESON_BUILD}" setup
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        -Dva-libs-path="/lib"
        -Dallow-fallback-for=libva
        -Dforce_fallback_for=libva
        -Degl=disabled
        -Dgallium-drivers=d3d12
        -Dgallium-va=enabled
        -Dllvm=disabled
        -Dopengl=false
        -Dvideo-codecs=all
        -Dvulkan-drivers=
        -Dlibva:with_glx=no
        -Dlibva:with_wayland=no
        -Dlibva:with_x11=no
        -Dspirv-tools=disabled
        -Dzlib=disabled
        -Dzstd=disabled
        -Dexpat=disabled
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${PATCH_SOURCE_DIR}"
  COMMAND
    "${MESON_BUILD}" compile --verbose
  COMMAND
    "${CMAKE_COMMAND}" -E env "DESTDIR=${CMAKE_INSTALL_PREFIX}" --
      "${MESON_BUILD}" install
  COMMAND
    # DirectX-Headers / d3dx12 are build-time only dependencies of mesa,
    # statically linked into vaon12_drv_video.dll. Remove them from the staged
    # install so they are not shipped in the dist tree or artifacts.
    "${CMAKE_COMMAND}" -E rm -rf --
      "${CMAKE_INSTALL_PREFIX}/include/directx"
      "${CMAKE_INSTALL_PREFIX}/include/dxguids"
      "${CMAKE_INSTALL_PREFIX}/include/wsl"
      "${CMAKE_INSTALL_PREFIX}/include/composition"
      "${CMAKE_INSTALL_PREFIX}/lib/libDirectX-Guids.a"
      "${CMAKE_INSTALL_PREFIX}/lib/libd3dx12-format-properties.a"
)
