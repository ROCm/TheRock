# AOCL-BLAS integration for TheRock
# Builds the BLAS component of AOCL 5.2 from source with ILP64 support
# Provides BLIS/BLAS/CBLAS functions as CPU reference BLAS for rocBLAS clients/tests
#
# This replaces OpenBLAS as an alternative CPU BLAS provider for testing.
# The resulting library is named libaocl.a (AOCL's unified naming), but only
# contains the BLAS component (controlled by -DENABLE_AOCL_BLAS=ON).
#
# This project follows the host-blas pattern:
#   - Builds static library (AOCL preference for ILP64)
#   - Installs to lib/host-math/ (alongside OpenBLAS)
#   - Provides AOCL CMake package for downstream consumption
#   - Only built when THEROCK_BUILD_TESTING=ON (via HOST_MATH feature group)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    # Fetch AOCL 5.2 from GitHub
    # AOCL-BLAS 5.2 provides ILP64 BLAS support needed for rocBLAS stress tests
    therock_subproject_fetch(therock-aocl-blas-sources
      CMAKE_PROJECT
      SOURCE_DIR "${_source_dir}"
      GIT_REPOSITORY https://github.com/amd/aocl.git
      GIT_TAG AOCL-5.2
      GIT_SHALLOW TRUE
      TOUCH "${_download_stamp}"
    )

    # AOCL-BLAS CMake configuration matching install.sh behavior
    # See: /root/coding/rocm-libraries/projects/rocblas/install.sh lines 316-317
    therock_cmake_subproject_declare(therock-aocl-blas
      USE_DIST_AMDGPU_TARGETS
      BACKGROUND_BUILD
      EXCLUDE_FROM_ALL
      NO_MERGE_COMPILE_COMMANDS
      OUTPUT_ON_FAILURE
      EXTERNAL_SOURCE_DIR .
      INTERFACE_LINK_DIRS "lib/host-math/lib"
      # RPATH logic needs to know that executables/libs for this project are in
      # a non-default location (same as OpenBLAS).
      INSTALL_RPATH_EXECUTABLE_DIR "lib/host-math/bin"
      INSTALL_RPATH_LIBRARY_DIR "lib/host-math/lib"
      INTERFACE_INSTALL_RPATH_DIRS "lib/host-math/lib"
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        # Build static library (AOCL's CMake builds static by default for ILP64)
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=Release
        # Enable ILP64 support (64-bit integers) - required for rocBLAS stress tests
        -DENABLE_ILP64=ON
        # Enable AOCL BLAS and utilities
        # Note: Output is named libaocl.a (AOCL's unified naming convention),
        # but only contains BLAS component due to selective enabling below
        -DENABLE_AOCL_BLAS=ON
        -DENABLE_AOCL_UTILS=ON
        # Disable LAPACK (not needed for rocBLAS clients - we only need BLAS)
        -DENABLE_AOCL_LAPACK=OFF
        # Enable multithreading support for parallel BLAS operations
        -DENABLE_MULTITHREADING=ON
        # Disable OpenMP library linking (let AOCL handle it internally)
        -DOpenMP_libomp_LIBRARY=""
        # Set install directories to match TheRock's host-math layout
        -DCMAKE_INSTALL_BINDIR=lib/host-math/bin
        -DCMAKE_INSTALL_LIBDIR=lib/host-math/lib
        -DCMAKE_INSTALL_INCLUDEDIR=lib/host-math/include
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    
    # Register AOCL-BLAS package for downstream consumption
    # This makes find_package(aocl) work in dependent projects (package name stays "aocl")
    therock_cmake_subproject_provide_package(therock-aocl-blas aocl lib/host-math/lib/cmake/aocl)
    therock_cmake_subproject_activate(therock-aocl-blas)

    # Validate that the built BLAS library exists
    therock_test_validate_static_lib(
      PATH dist/lib/host-math/lib
      LIB_NAMES libaocl.a
    )
    
    # Copy headers from dist/include to dist/lib/host-math/include
    # AOCL's build system ignores CMAKE_INSTALL_INCLUDEDIR, so we copy them manually
    # This must run after the stage install is complete
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/dist/lib/host-math/include/blis.h"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/dist/lib/host-math/include"
      COMMAND ${CMAKE_COMMAND} -E copy_directory
              "${CMAKE_CURRENT_BINARY_DIR}/dist/include"
              "${CMAKE_CURRENT_BINARY_DIR}/dist/lib/host-math/include"
      DEPENDS therock-aocl-blas+stage
      COMMENT "Copying AOCL-BLAS headers to lib/host-math/include/"
    )
    add_custom_target(therock-aocl-blas-headers ALL
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/dist/lib/host-math/include/blis.h"
    )
    
    therock_provide_artifact(host-aocl-blas
      DESCRIPTOR artifact-aocl.toml
      TARGET_NEUTRAL
      COMPONENTS
        dbg
        dev
        doc
        lib
      SUBPROJECT_DEPS therock-aocl-blas
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(AOCL_BUILD)

# Install to lib/host-math to keep AOCL alongside OpenBLAS
# This creates a unified location for all host math libraries
set(CMAKE_INSTALL_BINDIR "lib/host-math/bin")
set(CMAKE_INSTALL_LIBDIR "lib/host-math/lib")
set(CMAKE_INSTALL_INCLUDEDIR "lib/host-math/include")

# Build AOCL from the fetched source
add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/build-aocl)

# AOCL's CMake should create the aocl target
# The library will be named libaocl.a by default
# However, AOCL's build system installs to lib/ and include/ instead of lib/host-math/
# So we manually install both to the correct locations
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/build-aocl/libaocl.a"
        DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Install BLIS/BLAS headers
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/build-aocl/blis/install_package/include/"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

# Install our custom CMake package config
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/aocl-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/aocl-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/aocl-config.cmake" 
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aocl)
