therock_subproject_fetch(therock-iree-sources
  CMAKE_PROJECT
  SOURCE_DIR "/home/astgeorg/Dev/c++/iree"
)

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
  EXTERNAL_SOURCE_DIR "/home/astgeorg/Dev/c++/iree"
  BUILD_DEPS
    rocm-cmake
    hip-clr
  RUNTIME_DEPS
    hip-clr
  INSTALL_COMPONENTS
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
    -DIREE_HIP_USE_SYSTEM_PACKAGE=ON
)
therock_cmake_subproject_provide_package(therock-iree-runtime IREERuntime lib/cmake/IREE)
therock_cmake_subproject_activate(therock-iree-runtime)

add_dependencies(therock-third-party therock-iree-runtime)
