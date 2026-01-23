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

    # AOCL-BLAS CMake configuration matching rocBLAS install.sh behavior
    # See: https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/install.sh#L316
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
        # Let AOCL's build system discover and link the appropriate OpenMP implementation.
        # By setting OpenMP_libomp_LIBRARY empty, we avoid forcing a specific OpenMP library
        # from this wrapper project, allowing AOCL to manage OpenMP discovery internally.
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

# Install BLIS/BLAS headers to lib/host-math/include/aocl/
# AOCL installs headers to ${CMAKE_INSTALL_PREFIX}/include, so we copy them to our desired location
# We use an aocl/ subdirectory to match OpenBLAS's pattern (openblas/ subdirectory)
#
# Note: GLOB_RECURSE evaluates at install time (not configure time). If AOCL's build changes
# and stops generating expected headers, this won't be detected until install. The install
# will print a status message showing how many headers were copied, which serves as verification.
# We also validate that critical AOCL-BLAS headers (blis.h, cblas.h, blis64.h) are present.
install(CODE "
  file(GLOB_RECURSE AOCL_HEADERS \"\${CMAKE_INSTALL_PREFIX}/include/*.h\" \"\${CMAKE_INSTALL_PREFIX}/include/*.hh\")
  list(LENGTH AOCL_HEADERS NUM_HEADERS)
  if(NUM_HEADERS EQUAL 0)
    message(FATAL_ERROR \"No AOCL headers found in \${CMAKE_INSTALL_PREFIX}/include/\")
  endif()
  
  # Validate critical AOCL-BLAS headers are present
  set(CRITICAL_HEADERS blis.h cblas.h blis64.h)
  foreach(CRITICAL_HEADER \${CRITICAL_HEADERS})
    set(FOUND_CRITICAL FALSE)
    foreach(HEADER \${AOCL_HEADERS})
      get_filename_component(HEADER_NAME \"\${HEADER}\" NAME)
      if(HEADER_NAME STREQUAL \"\${CRITICAL_HEADER}\")
        set(FOUND_CRITICAL TRUE)
        break()
      endif()
    endforeach()
    if(NOT FOUND_CRITICAL)
      message(FATAL_ERROR \"Critical AOCL header \${CRITICAL_HEADER} not found in \${CMAKE_INSTALL_PREFIX}/include/\")
    endif()
  endforeach()
  
  file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aocl\")
  foreach(HEADER \${AOCL_HEADERS})
    file(RELATIVE_PATH REL_PATH \"\${CMAKE_INSTALL_PREFIX}/include\" \"\${HEADER}\")
    get_filename_component(HEADER_DIR \"\${REL_PATH}\" DIRECTORY)
    if(HEADER_DIR)
      file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aocl/\${HEADER_DIR}\")
    endif()
    file(COPY_FILE \"\${HEADER}\" \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aocl/\${REL_PATH}\")
  endforeach()
  message(STATUS \"Copied \${NUM_HEADERS} AOCL headers to ${CMAKE_INSTALL_INCLUDEDIR}/aocl\")
")

# Install our custom CMake package config
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/aocl-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/aocl-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/aocl-config.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aocl)
