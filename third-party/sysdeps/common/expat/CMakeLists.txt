if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-expat-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://github.com/libexpat/libexpat/releases/download/R_2_7_3/expat-2.7.3.tar.xz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/expat-2.7.3.tar.xz"
      URL_HASH "SHA512=d2e495a9a2b902d38d838ec514da67ff56b76cdefcf89d0c320e2f319a35e439bd697880334d0ee70ca79b4515ac94c0c9b9ff97f899bcaa89308f1c8efaee0d"
      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-expat
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      FPRINT_SOURCE_DIR "${_source_dir}"
      FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
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
    therock_cmake_subproject_provide_package(therock-expat expat lib/rocm_sysdeps/lib/cmake/expat)
    therock_cmake_subproject_activate(therock-expat)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libexpat.so
    )

    add_test(
      NAME expat_header_test_1
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/expat.h"
    )

    add_test(
      NAME expat_header_test_2
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/expat_config.h"
    )

    add_test(
      NAME expat_header_test_3
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/expat_external.h"
    )

   return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(EXPAT_BUILD VERSION "2.7.3")

# Do builds from a copy of the source directory, in case we need to
# patch the sources.
set(SOURCE_COPY_DIR "${CMAKE_CURRENT_BINARY_DIR}/s")

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${SOURCE_COPY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${SOURCE_COPY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
      # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
      # then escaped single quotes to make it to the linker command line.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      "CPPFLAGS=${EXTRA_CPPFLAGS}"
      --
    "${SOURCE_COPY_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      # Disable examples and tests.
      --without-examples
      --without-tests
      # Disable maintainer-mode, otherwise our copy of the sources directory
      # will potentially alter the timestamp of acinclude.m4, which will
      # trigger a reconfigure with a version of aclocal we do not have.
      --disable-maintainer-mode
  COMMAND
    make V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/expat-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/expat-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/expat-config.cmake" DESTINATION lib/cmake/expat)
