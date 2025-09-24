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
      # We don't have an lzma sysdep yet and may not need it.
      --without-lzma
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
