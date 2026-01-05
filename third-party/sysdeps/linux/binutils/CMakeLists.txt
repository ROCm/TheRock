if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-binutils-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "http://ftpmirror.gnu.org/gnu/binutils/binutils-2.45.tar.gz"
      URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/binutils-2.45.tar.gz
      URL_HASH "SHA512=5c71ec80884c0b4c0f7a4f8600946f1f6feebe584261b63185b0942cba4062d5b6cf2337539132d0ca03505c0a74f3c2760bc62ed08843c3b7db03df7f5798ad"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-binutils
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

    therock_cmake_subproject_provide_package(therock-binutils LibIberty lib/rocm_sysdeps/lib/cmake/libiberty)
    therock_cmake_subproject_activate(therock-binutils)

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(BINUTILS_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

set(EXTRA_CPPFLAGS)
string(APPEND EXTRA_CPPFLAGS " -fPIC")
message(STATUS "EXTRA_CPPFLAGS=${EXTRA_CPPFLAGS}")

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # We have to patch the sources so make a fresh copy.
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "CFLAGS=${EXTRA_CPPFLAGS}"
      "CPPFLAGS=${EXTRA_CPPFLAGS}"
      --
    "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --enable-install-libiberty
      --disable-shared
      --enable-static
      --libdir="${CMAKE_INSTALL_PREFIX}/lib"
      --includedir="${CMAKE_INSTALL_PREFIX}/include"
      --disable-nls
      --disable-werror
  COMMAND
    make -j 1 V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}

  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh"
)

# This matches the conventions of the FindLibIberty.cmake module.
# Other than that, there is so source of truth for this.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libiberty-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libiberty-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libiberty-config.cmake" DESTINATION lib/cmake/libiberty)
