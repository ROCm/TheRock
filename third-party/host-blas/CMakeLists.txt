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
      URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/OpenBLAS-0.3.31.tar.gz
      URL_HASH SHA256=6dd2a63ac9d32643b7cc636eab57bf4e57d0ed1fff926dfbc5d3d97f2d2be3a6
      # Originally posted MD5 was recomputed as SHA256 manually:
      # URL_HASH MD5=853a0c5c0747c5943e7ef4bbb793162d
      TOUCH "${_download_stamp}"
    )

    # MSVC does not support VLA and AT&T assembly which are required by DYNAMIC_ARCH
    # See: https://github.com/OpenMathLib/OpenBLAS/issues/2826#issuecomment-814170090
    # Multi-arch support on Windows seems to be a problem not limited to MSVC
    # See: https://github.com/OpenMathLib/OpenBLAS/issues/3985
    if(WIN32)
      set(_enable_dynamicArch "OFF")
    else()
      set(_enable_dynamicArch "ON")
    endif()

    ##############################################################################
    # LP64 OpenBLAS
    ##############################################################################
    set(_host_blas_subproject_names)
    therock_cmake_subproject_declare(therock-host-blas
      USE_DIST_AMDGPU_TARGETS
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/host-blas"
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
        # https://github.com/OpenMathLib/OpenBLAS?tab=readme-ov-file#x86x86-64
        # Picking HASWELL as a baseline that supports modern AMD and Intel CPUs
        -DTARGET=HASWELL
        -DDYNAMIC_ARCH=${_enable_dynamicArch}
        -DBUILD_TESTING=OFF
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-host-blas OpenBLAS lib/host-math/lib/cmake/OpenBLAS)
    therock_cmake_subproject_provide_package(therock-host-blas cblas lib/host-math/lib/cmake/OpenBLAS)
    therock_cmake_subproject_activate(therock-host-blas)

    therock_test_validate_shared_lib(
      PATH host-blas/dist/lib/host-math/lib
      LIB_NAMES librocm-openblas.so
    )

    list(APPEND _host_blas_subproject_names therock-host-blas)

    ##############################################################################
    # ILP64 OpenBLAS
    ##############################################################################
    therock_cmake_subproject_declare(therock-host-blas64
      USE_DIST_AMDGPU_TARGETS
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/host-blas64"
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
        # https://github.com/OpenMathLib/OpenBLAS?tab=readme-ov-file#x86x86-64
        # Picking HASWELL as a baseline that supports modern AMD and Intel CPUs
        -DTARGET=HASWELL
        -DDYNAMIC_ARCH=${_enable_dynamicArch}
        -DBUILD_TESTING=OFF
        -DINTERFACE64=1
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-host-blas64 OpenBLAS64 lib/host-math/lib/cmake/OpenBLAS64)
    therock_cmake_subproject_provide_package(therock-host-blas64 cblas64 lib/host-math/lib/cmake/OpenBLAS64)
    therock_cmake_subproject_activate(therock-host-blas64)

    therock_test_validate_shared_lib(
      PATH host-blas64/dist/lib/host-math/lib
      LIB_NAMES librocm-openblas64.so
    )

    list(APPEND _host_blas_subproject_names therock-host-blas64)

    therock_provide_artifact(host-blas
      DESCRIPTOR artifact-host-OpenBLAS.toml
      TARGET_NEUTRAL
      COMPONENTS
        dbg
        dev
        doc
        lib
        run
      SUBPROJECT_DEPS ${_host_blas_subproject_names}
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(OpenBLAS_BUILD)

if(INTERFACE64)
  set(_SUFFIX64 64)
  set(_SUFFIX64_UNDERSCORE _64)
endif()

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
set_target_properties(openblas${_SUFFIX64_UNDERSCORE}_shared PROPERTIES
  # Private SONAME/DLL name.
  OUTPUT_NAME rocm-openblas${_SUFFIX64}
  # Install location in the library is hard-coded to include in the install root.
  INTERFACE_INCLUDE_DIRECTORIES $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/openblas${_SUFFIX64}>
)

# Some of the test targets do not build with an altered library name on
# Windows. Just uniformly exclude them from the build.
set_target_properties(openblas_utest PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(openblas_utest_ext PROPERTIES EXCLUDE_FROM_ALL TRUE)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cblas-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cblas${_SUFFIX64}-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cblas${_SUFFIX64}-config.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenBLAS${_SUFFIX64})
