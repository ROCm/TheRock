# Set up install directories for IREE
# TheRock already sets CMAKE_INSTALL_LIBDIR="lib" to prevent lib64
# We include GNUInstallDirs to get CMAKE_INSTALL_INCLUDEDIR and CMAKE_INSTALL_BINDIR
# This follows the same pattern as ROCm projects (see ROCMInstallTargets.cmake)
include(GNUInstallDirs)

# Get HIP headers path from hip-clr dist directory
get_target_property(_hip_clr_dist_dir hip-clr THEROCK_DIST_DIR)

therock_cmake_subproject_declare(therock-iree-runtime
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/iree-runtime"
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree"
  BUILD_DEPS
    rocm-cmake
    hip-clr
  RUNTIME_DEPS
    hip-clr
    rocminfo
  INSTALL_OPTIONAL_COMPONENTS
    IREEDevLibraries-Runtime
    IREECMakeExports
  CMAKE_ARGS
    -DHIP_PLATFORM=amd
    # Clears a ROCM_PATH/DIR environment variable, preventing the build
    # from accidentally finding system ROCm instead of TheRock's ROCm.
    -DROCM_PATH=
    -DROCM_DIR=
    # Pass install directories to IREE (CMAKE_INSTALL_LIBDIR comes from toolchain)
    -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
    -DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
    -DIREE_VISIBILITY_HIDDEN=OFF
    -DIREE_BUILD_COMPILER=OFF
    -DIREE_BUILD_TESTS=OFF
    -DIREE_BUILD_SAMPLES=OFF
    -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
    -DIREE_HAL_DRIVER_DEFAULTS=OFF
    -DIREE_HAL_DRIVER_HIP=ON
    -DIREE_HIP_TEST_TARGET_CHIP=gfx942
    -DIREE_ENABLE_THREADING=ON
    -DIREE_BUILD_TESTS=OFF
    -DIREE_USE_SYSTEM_DEPS=ON
)
therock_cmake_subproject_provide_package(therock-iree-runtime IREERuntime lib/cmake/IREE)
therock_cmake_subproject_activate(therock-iree-runtime)

add_dependencies(therock-third-party therock-iree-runtime)

# Fusilli is only built when the Fusilli plugin is enabled
if(THEROCK_ENABLE_FUSILLI_PLUGIN)
  therock_cmake_subproject_declare(therock-fusilli
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilli"
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli"
    BUILD_DEPS
      therock-iree-runtime
      therock-catch2
    CMAKE_ARGS
      -DFUSILLI_BUILD_TESTS=OFF
      -DFUSILLI_BUILD_BENCHMARKS=OFF
      -DFUSILLI_SYSTEMS_AMDGPU=ON
      -DFUSILLI_CODE_COVERAGE=OFF
      -DFUSILLI_ENABLE_LOGGING=OFF
      -DFUSILLI_ENABLE_CLANG_TIDY=OFF
  )
  therock_cmake_subproject_provide_package(therock-fusilli Fusilli lib/cmake/fusilli)
  therock_cmake_subproject_activate(therock-fusilli)

  add_dependencies(therock-third-party therock-fusilli)

  ##############################################################################
  # fusilli hipDNN plugin
  ##############################################################################

  therock_cmake_subproject_declare(fusilli_plugin
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli/plugins/hipdnn-plugin"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilli_plugin"
    BACKGROUND_BUILD
    CMAKE_ARGS
      "-DFUSILLI_PLUGIN_SKIP_TESTS=$<NOT:$<BOOL:${THEROCK_BUILD_TESTING}>>"
    CMAKE_INCLUDES
      therock_explicit_finders.cmake # TODO: is this necessary? I have no idea what this does.
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipDNN
      therock-googletest
      therock-flatbuffers
      therock-nlohmann-json
      therock-fmt
      therock-spdlog
      therock-iree-runtime
      therock-fusilli
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(fusilli_plugin
  SUBDIRS
    .
  )
  therock_cmake_subproject_activate(fusilli_plugin)

  therock_provide_artifact(fusilli-plugin
    DESCRIPTOR artifact-fusilli-plugin.toml
    COMPONENTS
      dbg
      lib
    SUBPROJECT_DEPS
      fusilli_plugin
  )
endif(THEROCK_ENABLE_FUSILLI_PLUGIN)
