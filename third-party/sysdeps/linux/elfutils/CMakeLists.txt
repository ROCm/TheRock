if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-elfutils-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://sourceware.org/elfutils/ftp/0.192/elfutils-0.192.tar.bz2"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/elfutils-0.192.tar.bz2"
      URL_HASH "SHA512=543188f5f2cfe5bc7955a878416c5f252edff9926754e5de0c6c57b132f21d9285c9b29e41281e93baad11d4ae7efbbf93580c114579c182103565fe99bd3909"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-elfutils
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      RUNTIME_DEPS
        therock-bzip2
        therock-liblzma
        therock-zlib
        therock-zstd
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
    therock_cmake_subproject_provide_package(therock-elfutils LibElf lib/rocm_sysdeps/lib/cmake/LibElf)
    therock_cmake_subproject_provide_package(therock-elfutils libdw lib/rocm_sysdeps/lib/cmake/libdw)
    therock_cmake_subproject_activate(therock-elfutils)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES libelf.so libdw.so libasm.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(ELFUTILS_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# HACK: elfutils consults pkg-config for cflags for the compression libraries
# but then does not seem to use them everywhere needed. So we just hard-code
# paths.
set(EXTRA_CPPFLAGS)
string(APPEND EXTRA_CPPFLAGS " -I${THEROCK_BINARY_DIR}/third-party/sysdeps/linux/bzip2/build/stage/lib/rocm_sysdeps/include")
string(APPEND EXTRA_CPPFLAGS " -I${THEROCK_BINARY_DIR}/third-party/sysdeps/linux/liblzma/build/stage/lib/rocm_sysdeps/include")
string(APPEND EXTRA_CPPFLAGS " -I${THEROCK_BINARY_DIR}/third-party/sysdeps/linux/zlib/build/stage/lib/rocm_sysdeps/include")
string(APPEND EXTRA_CPPFLAGS " -I${THEROCK_BINARY_DIR}/third-party/sysdeps/linux/zstd/build/stage/lib/rocm_sysdeps/include")
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
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # CMake copy directory does not preserve timestamps. If the stamp-po is
    # out of date, the build will try to regenerate docs, which requires a bunch
    # of extra tools and fails mysteriously if not present with something like:
    #   mv: cannot stat 't-de.gmo': No such file or directory
    # Ensuring it has the most current timestamp post copy/patch disables doc
    # regneration.
    "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/s/po/stamp-po"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
      # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
      # then escaped single quotes to make it to the linker command line.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\'"
      "CPPFLAGS=${EXTRA_CPPFLAGS}"
      --
    "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --disable-debuginfod
      --enable-install-elfh
      --with-bzlib --with-zlib --with-zstd
      --with-lzma
  COMMAND
    make -j "${PAR_JOBS}" V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}

  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh"
)

# This matches the conventions of the FindLibElf.cmake module in ROCR-Runtime.
# Other than that, there is so source of truth for this.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libelf-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libelf-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libelf-config.cmake" DESTINATION lib/cmake/LibElf)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libdw-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libdw-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libdw-config.cmake" DESTINATION lib/cmake/libdw)
