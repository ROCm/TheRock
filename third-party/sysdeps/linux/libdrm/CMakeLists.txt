if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libdrm-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://gitlab.freedesktop.org/mesa/drm/-/archive/libdrm-2.4.124/drm-libdrm-2.4.124.tar.bz2"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/drm-libdrm-2.4.124.tar.bz2"
      URL_HASH "SHA256=18e66044e0542040614a7904b6a2c0e5249a81e705fe9ba5a1cc2e5df11416e6"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-libdrm
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
    therock_cmake_subproject_activate(therock-libdrm)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libdrm.so libdrm_amdgpu.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBDRM_BUILD)

if(NOT PATCHELF OR NOT MESON_BUILD)
  message(FATAL_ERROR "Missing PATCHELF or MESON_BUILD from super-project")
endif()

# Meson refuses to build if the source dir is a subdir of the build dir, so
# make it a sibling.
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
    # Inline the amdgpu device table.
    "${CMAKE_CURRENT_SOURCE_DIR}/inline_amdgpu_ids.sh" "${PATCH_SOURCE_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E chdir "${PATCH_SOURCE_DIR}"
    "${CMAKE_COMMAND}" -E env
      # Escaping hack: experimentally determined to persist through the layers.
      "LDFLAGS=-Wl,-rpath='$$ORIGIN' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      --
      "${MESON_BUILD}" setup "${CMAKE_CURRENT_BINARY_DIR}"
        --reconfigure
        # We generate relocatable, arch neutral directory layouts and use
        # DESTDIR to install int ot the lib/rocm_sysdeps tree.
        --prefix "/"
        -Dpkgconfig.relocatable=true
        -Dlibdir=lib
        # Only enable the libraries we want.
        -Damdgpu=enabled
        -Dintel=disabled
        -Dman-pages=disabled
        -Dnouveau=disabled
        -Dradeon=disabled
        -Dvmwgfx=disabled
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
