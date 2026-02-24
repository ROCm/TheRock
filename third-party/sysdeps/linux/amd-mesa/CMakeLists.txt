if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # setup submodule project
    set(_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/mesa-fork")

    therock_cmake_subproject_declare(therock-amd-mesa
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
        "-DMESON_BUILD=${MESON_BUILD}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      RUNTIME_DEPS
        therock-libdrm
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
      LIB_NAMES libgallium_drv_video.so radeonsi_drv_video.so libva.so libva-drm.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(MESA_BUILD)
if(NOT PATCHELF OR NOT MESON_BUILD)
  message(FATAL_ERROR "Missing PATCHELF or MESON_BUILD from super-project")
endif()

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
    # Step 1: Initial meson setup to download libva-2.22.0 subproject
    # This is required before patching because patch_source.sh modifies files in the libva subproject
    "${CMAKE_COMMAND}" -E env
      "LDFLAGS=-Wl,-rpath='$$ORIGIN' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      --
      "${MESON_BUILD}" setup
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        -Dva-libs-path="/lib"
        -Dallow-fallback-for=libva # NOTE: Use local build of VA library
        -Dforce_fallback_for=libva # NOTE: Enforce local build, if public VA found
        -Dpkg_config_path="${CMAKE_CURRENT_BINARY_DIR}"/../../../libdrm/build/dist/lib/rocm_sysdeps/lib/pkgconfig
        -Damd-decode-only=true # NOTE: enable decode only for TheRock
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 2: Patch source files including libva driver path configuration
    # Modifies library names and driver search paths for ROCm sysdeps integration
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${PATCH_SOURCE_DIR}"
  COMMAND
    # Step 3: Reconfigure meson to build from patched source
    # This ensures the build system uses the modified source files
    "${CMAKE_COMMAND}" -E env
      "LDFLAGS=-Wl,-rpath='$$ORIGIN' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      --
      "${MESON_BUILD}" setup
        --reconfigure
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        -Dva-libs-path="/lib"
        -Dallow-fallback-for=libva # NOTE: Use local build of VA library
        -Dforce_fallback_for=libva # NOTE: Enforce local build, if public VA found
        -Dpkg_config_path="${CMAKE_CURRENT_BINARY_DIR}"/../../../libdrm/build/dist/lib/rocm_sysdeps/lib/pkgconfig
        -Damd-decode-only=true # NOTE: enable decode only for TheRock
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${PATCH_SOURCE_DIR}"
  COMMAND
    "${MESON_BUILD}" compile --verbose
  COMMAND
    "${CMAKE_COMMAND}" -E env "DESTDIR=${CMAKE_INSTALL_PREFIX}" --
      "${MESON_BUILD}" install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
)
