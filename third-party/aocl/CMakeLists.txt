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

  # Fetch AOCL with Windows/TheRock integration fixes from tony-davis fork
  # TODO: Switch to official AMD repo once PRs are merged
  # AOCL-BLAS 5.2.1 provides ILP64 BLAS support with pinned submodule versions (reproducible builds)
  # Fixes:
  # 1. Use CMAKE_CURRENT_SOURCE_DIR instead of CMAKE_SOURCE_DIR for submodule paths
  # 2. Replace execute_process() anti-pattern with add_subdirectory() for BLIS build
  # 3. BLIS submodule points to tony-davis/blis with MSVC and OpenMP 2.0 fixes
  therock_subproject_fetch(therock-aocl-blas-sources
    CMAKE_PROJECT
    SOURCE_DIR "${_source_dir}"
    GIT_REPOSITORY https://github.com/tony-davis/aocl.git
    GIT_TAG windows-therock-integration
    GIT_SHALLOW TRUE
    TOUCH "${_download_stamp}"
  )

    # AOCL-BLAS CMake configuration matching rocBLAS install.sh behavior
    # See: https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/install.sh#L316
    #
  # Compiler Strategy (Windows):
  # - BLIS (core component of AOCL-BLAS) requires Clang for proper architecture flag handling
  # - On Windows: Uses Visual Studio's Clang-CL (from VS Developer PowerShell PATH)
  # - Clang-CL must be installed: "C++ Clang Compiler for Windows" VS component
  # - This matches AOCL's official Windows build method (see their README.md)
  
  # On Windows, use a toolchain file to force clang-cl before project() command
  # This ensures CMake detects clang-cl instead of falling back to cl.exe
  set(_aocl_cmake_args "-DSOURCE_DIR=${_source_dir}")
  if(WIN32)
    set(_toolchain_file "${CMAKE_CURRENT_SOURCE_DIR}/clang-cl-toolchain.cmake")
    list(APPEND _aocl_cmake_args "-DCMAKE_TOOLCHAIN_FILE=${_toolchain_file}")
    message(STATUS "Using clang-cl toolchain for AOCL: ${_toolchain_file}")
  endif()

  therock_cmake_subproject_declare(therock-aocl-blas
      USE_DIST_AMDGPU_TARGETS
      BACKGROUND_BUILD
      EXCLUDE_FROM_ALL
      NO_MERGE_COMPILE_COMMANDS
      OUTPUT_ON_FAILURE
      EXTERNAL_SOURCE_DIR .
      INTERFACE_LINK_DIRS "lib/host-math/${CMAKE_INSTALL_LIBDIR}"
      # RPATH logic needs to know that executables/libs for this project are in
      # a non-default location (same as OpenBLAS).
      INSTALL_RPATH_EXECUTABLE_DIR "lib/host-math/${CMAKE_INSTALL_BINDIR}"
      INSTALL_RPATH_LIBRARY_DIR "lib/host-math/${CMAKE_INSTALL_LIBDIR}"
      INTERFACE_INSTALL_RPATH_DIRS "lib/host-math/${CMAKE_INSTALL_LIBDIR}"
          CMAKE_ARGS
            ${_aocl_cmake_args}
            # Build static library (AOCL's CMake builds static by default for ILP64)
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=Release
        # Force Release configuration: AOCL's CMakeLists.txt defaults CMAKE_CONFIGURATION_TYPES to Debug
        # Without this, BLIS builds in Debug mode causing 100-450x slowdowns on trsm/trmm operations
        -DCMAKE_CONFIGURATION_TYPES=Release
        # Enable ILP64 support (64-bit integers) - required for rocBLAS stress tests
        -DENABLE_ILP64=ON
        # Enable AOCL BLAS only
        # Note: Output is named libaocl.a on Linux, aocl.lib on Windows (AOCL's unified naming convention),
        # We disable UTILS on Windows due to MSVC incompatibility issues with GCC-specific intrinsics
        -DENABLE_AOCL_BLAS=ON
        $<IF:$<PLATFORM_ID:Windows>,-DENABLE_AOCL_UTILS=OFF,-DENABLE_AOCL_UTILS=ON>
        # Disable LAPACK (not needed for rocBLAS clients - we only need BLAS)
        -DENABLE_AOCL_LAPACK=OFF
        # Enable multithreading support for parallel BLAS operations
        -DENABLE_MULTITHREADING=ON
        # Configure OpenMP for MSVC
        # MSVC's OpenMP support requires explicit flags and variables since FindOpenMP doesn't auto-detect
        # Note: MSVC OpenMP is built into vcruntime - no separate library needed (unlike GCC's libgomp)
        $<IF:$<PLATFORM_ID:Windows>,-DOpenMP_C_FLAGS=/openmp,-DOpenMP_C_FLAGS=-fopenmp>
        $<IF:$<PLATFORM_ID:Windows>,-DOpenMP_CXX_FLAGS=/openmp,-DOpenMP_CXX_FLAGS=-fopenmp>
        $<IF:$<PLATFORM_ID:Windows>,-DOpenMP_C_LIB_NAMES=,>
        $<IF:$<PLATFORM_ID:Windows>,-DOpenMP_CXX_LIB_NAMES=,>
        -DOpenMP_libomp_LIBRARY=""
        # Set install directories to match TheRock's host-math layout
        -DCMAKE_INSTALL_BINDIR=lib/host-math/bin
        -DCMAKE_INSTALL_LIBDIR=lib/host-math/lib
        -DCMAKE_INSTALL_INCLUDEDIR=lib/host-math/include
      EXTRA_DEPENDS
        "${_download_stamp}"
    )

    # Register AOCL package for downstream consumption
    # AOCL now properly exports AOCL::AOCL target via its own CMake package config
    therock_cmake_subproject_provide_package(therock-aocl-blas AOCL lib/host-math/${CMAKE_INSTALL_LIBDIR}/cmake/AOCL)
    therock_cmake_subproject_activate(therock-aocl-blas)

    # Validate that the built BLAS library exists
    # AOCL installs library with standard naming: libaocl.a on Linux, aocl.lib on Windows
    if(WIN32)
      set(_aocl_lib_name "aocl.lib")
    else()
      set(_aocl_lib_name "libaocl.a")
    endif()
    
    therock_test_validate_static_lib(
      PATH dist/lib/host-math/${CMAKE_INSTALL_LIBDIR}
      LIB_NAMES ${_aocl_lib_name}
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
# AOCL uses GNUInstallDirs, so we need to set the install dirs directly
# to avoid double-nesting (don't modify CMAKE_INSTALL_PREFIX, set the subdirs)
include(GNUInstallDirs)

# Override GNUInstallDirs defaults to install to lib/host-math subdirectory
set(CMAKE_INSTALL_BINDIR "lib/host-math/bin" CACHE PATH "Binary install dir" FORCE)
set(CMAKE_INSTALL_LIBDIR "lib/host-math/lib" CACHE PATH "Library install dir" FORCE)
set(CMAKE_INSTALL_INCLUDEDIR "lib/host-math/include" CACHE PATH "Include install dir" FORCE)

# Build AOCL from the fetched source
# AOCL now properly exports AOCL::AOCL target and installs to CMAKE_INSTALL_LIBDIR
add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/build-aocl)

# AOCL now installs its own proper CMake package config (AOCLConfig.cmake)
# to ${CMAKE_INSTALL_LIBDIR}/cmake/AOCL, so we don't need custom package files

# ============================================================================
# TODO: REMOVE THIS WORKAROUND once rocm-libraries PR for find_package(AOCL) is merged
# ============================================================================
# Temporary: Copy AOCL to legacy rocBLAS discovery location
# rocBLAS currently hardcodes a search path instead of using find_package(AOCL).
# This workaround maintains compatibility until rocm-libraries is updated.
#
# Legacy location: ${BUILD_DIR}/math-libs/BLAS/rocBLAS/build/deps/aocl/install_package/
# 
# Once the rocm-libraries PR is merged, rocBLAS will use:
#   find_package(AOCL REQUIRED)
#   target_link_libraries(rocblas-clients PRIVATE AOCL::AOCL)
install(CODE "
  # Construct path to rocBLAS build directory from AOCL's install prefix
  # CMAKE_INSTALL_PREFIX here is stage/, so we navigate to rocBLAS build
  set(_rocblas_deps_dir \"\${CMAKE_INSTALL_PREFIX}/../../math-libs/BLAS/rocBLAS/build/deps/aocl/install_package\")
  get_filename_component(_rocblas_deps_dir \"\${_rocblas_deps_dir}\" ABSOLUTE)
  
  # Only copy if rocBLAS build directory exists (might not exist in all build configs)
  if(EXISTS \"\${CMAKE_INSTALL_PREFIX}/../../math-libs/BLAS/rocBLAS\")
    # Create destination directories
    file(MAKE_DIRECTORY \"\${_rocblas_deps_dir}/lib\")
    file(MAKE_DIRECTORY \"\${_rocblas_deps_dir}/include\")
    
    # Copy library (platform-specific naming)
    if(WIN32)
      file(COPY \"\${CMAKE_INSTALL_PREFIX}/lib/host-math/lib/aocl.lib\"
           DESTINATION \"\${_rocblas_deps_dir}/lib\")
      message(STATUS \"[AOCL Workaround] Copied library for rocBLAS: \${_rocblas_deps_dir}/lib/aocl.lib\")
    else()
      file(COPY \"\${CMAKE_INSTALL_PREFIX}/lib/host-math/lib/libaocl.a\"
           DESTINATION \"\${_rocblas_deps_dir}/lib\")
      message(STATUS \"[AOCL Workaround] Copied library for rocBLAS: \${_rocblas_deps_dir}/lib/libaocl.a\")
    endif()
    
    # Copy headers (maintain directory structure)
    file(GLOB_RECURSE _aocl_headers \"\${CMAKE_INSTALL_PREFIX}/lib/host-math/include/*.h\")
    foreach(_header \${_aocl_headers})
      file(RELATIVE_PATH _rel_path \"\${CMAKE_INSTALL_PREFIX}/lib/host-math/include\" \"\${_header}\")
      get_filename_component(_header_dir \"\${_rel_path}\" DIRECTORY)
      if(_header_dir)
        file(MAKE_DIRECTORY \"\${_rocblas_deps_dir}/include/\${_header_dir}\")
      endif()
      file(COPY_FILE \"\${_header}\" \"\${_rocblas_deps_dir}/include/\${_rel_path}\")
    endforeach()
    message(STATUS \"[AOCL Workaround] Copied headers for rocBLAS: \${_rocblas_deps_dir}/include/\")
  else()
    message(STATUS \"[AOCL Workaround] rocBLAS build directory not found, skipping legacy copy\")
  endif()
")
# ============================================================================
# END TEMPORARY WORKAROUND
# ============================================================================
