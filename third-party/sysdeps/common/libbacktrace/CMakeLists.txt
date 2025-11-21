if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-libbacktrace-sources
      SOURCE_DIR "${_source_dir}"
      # Original project location: https://github.com/ianlancetaylor/libbacktrace/
      # Revision used: b9e40069c0b47a722286b94eb5231f7f05c08713 of master
      # branch.
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libbacktrace-b9e40069c0b47a722286b94eb5231f7f05c08713.zip"
      URL_HASH "SHA512=8acc3324d367413f41a7a65d86b96ad536907ef8c54905041670f5b9adac229a45d454a954385f230c4236d0d69d486551194b1298694af9df2229a05e998997"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-libbacktrace
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
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
    therock_cmake_subproject_provide_package(therock-libbacktrace libbacktrace lib/rocm_sysdeps/lib/cmake/libbacktrace)
    therock_cmake_subproject_activate(therock-libbacktrace)

    add_test(
      NAME libbacktrace_archive_test
      COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/lib/libbacktrace.a"
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBBACKTRACE_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
      # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
      # then escaped single quotes to make it to the linker command line.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN\\'"
      "CPPFLAGS=${EXTRA_CPPFLAGS} -fPIC"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    make -j "${PAR_JOBS}" V=1
  COMMAND
    make install
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libbacktrace-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace-config.cmake" DESTINATION lib/cmake/libbacktrace)
