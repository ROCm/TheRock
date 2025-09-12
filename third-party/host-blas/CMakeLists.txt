# Right now we only support OpenBLAS as the host BLAS library.
# This will be extended later, including allowing to use the system BLAS of
# your choice.

# Note that this project is a bit unique in third-party:
#   a. It builds shared on both Linux and Windows. Most of our third-party
#      deps, we build static on Windows and use symbol/soname alterations on
#      Linux to make them cooperate.
#   b. It populates the lib/host-math directory instead of the root. This
#      is a self contained prefix.
#   c. On Windows, it puts runtime files (i.e. DLLs) in the root install bin/
#      directory. On Windows, all of the DLLs go in the same location, and
#      only development files are scoped to lib/host-math.
#   d. It publishes its own cblas package, delegating to the OpenBLAS::OpenBLAS
#      library.

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-OpenBLAS-sources
      CMAKE_PROJECT
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz
      URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/OpenBLAS-0.3.30.tar.gz
      URL_HASH SHA256=27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d
      # Originally posted MD5 was recomputed as SHA256 manually:
      # URL_HASH MD5=853a0c5c0747c5943e7ef4bbb793162d
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-host-blas
      BACKGROUND_BUILD
      EXCLUDE_FROM_ALL
      NO_MERGE_COMPILE_COMMANDS
      OUTPUT_ON_FAILURE
      EXTERNAL_SOURCE_DIR .
      INTERFACE_LINK_DIRS "lib/host-math/lib"
      # RPATH logic needs to know that executables/libs for this project are in
      # a non-default location.
      INSTALL_RPATH_EXECUTABLE_DIR "lib/host-math/bin"
      INSTALL_RPATH_LIBRARY_DIR "lib/host-math/lib"
      INTERFACE_INSTALL_RPATH_DIRS "lib/host-math/lib"
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        -DBUILD_SHARED_LIBS=ON
        # TODO: DYNAMIC_ARCH=ON produces illegal elf files
        # See: https://github.com/ROCm/TheRock/issues/83
        -DDYNAMIC_ARCH=OFF
        -DC_LAPACK=ON
        -DBUILD_TESTING=OFF
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-host-blas OpenBLAS lib/host-math/lib/cmake/OpenBLAS)
    therock_cmake_subproject_provide_package(therock-host-blas cblas lib/host-math/lib/cmake/OpenBLAS)
    therock_cmake_subproject_activate(therock-host-blas)

    therock_test_validate_shared_lib(
      PATH dist/lib/host-math/lib
      LIB_NAMES librocm-openblas.so
    )

    therock_provide_artifact(host-blas
      DESCRIPTOR artifact-host-OpenBLAS.toml
      TARGET_NEUTRAL
      COMPONENTS
        dbg
        dev
        doc
        lib
        run
      SUBPROJECT_DEPS therock-host-blas
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(OpenBLAS_BUILD)

# Note that we install to the top-level but want our development assets and Posix
# runfiles to install to lib/host-math. Since we are building shared, on Windows,
# we want the DLL to go to the root bin/ dir so that it can be dynamically
# loaded by ROCm binaries. This is a bit of an abuse.
if(WIN32)
  set(CMAKE_INSTALL_BINDIR "bin")
else()
  set(CMAKE_INSTALL_BINDIR "lib/host-math/bin")
endif()
set(CMAKE_INSTALL_LIBDIR "lib/host-math/lib")
set(CMAKE_INSTALL_INCLUDEDIR "lib/host-math/include")

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/build-openblas)

# Alterations.
# We would like this to be a library scoped to ROCm, so give it a custom
# name.
# TODO: Also set Linux symbol versioning flags.
set_target_properties(openblas_shared PROPERTIES
  # Private SONAME/DLL name.
  OUTPUT_NAME rocm-openblas
  # Install location in the library is hard-coded to include in the install root.
  INTERFACE_INCLUDE_DIRECTORIES $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/openblas>
)

# Some of the test targets do not build with an altered library name on
# Windows. Just uniformly exclude them from the build.
set_target_properties(openblas_utest PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(openblas_utest_ext PROPERTIES EXCLUDE_FROM_ALL TRUE)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cblas-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cblas-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cblas-config.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenBLAS)
