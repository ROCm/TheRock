# Right now we only support OpenBLAS as the host BLAS library.
# This will be extended later, including allowing to use the system BLAS of
# your choice.

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-OpenBLAS-sources
      CMAKE_PROJECT
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz
      URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/OpenBLAS-0.3.29.tar.gz
      URL_HASH SHA256=38240eee1b29e2bde47ebb5d61160207dc68668a54cac62c076bb5032013b1eb
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
      INSTALL_DESTINATION "lib/host-math"
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
      LIB_NAMES libopenblas.so
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

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/build-openblas)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cblas-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cblas-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cblas-config.cmake" DESTINATION lib/cmake/OpenBLAS)
