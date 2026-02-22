therock_subproject_fetch(therock-eigen-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/eigen-3.4.0.tar.bz2
  URL_HASH SHA256=b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626
)


# Note: not using
#   COMPILER_TOOLCHAIN
#     amd-llvm
# below due to these errors (at least on Windows)
#   [build] [therock-eigen configure] -- Performing Test standard_math_library_linked_to_automatically
#   [build] [therock-eigen configure] -- Performing Test standard_math_library_linked_to_automatically - Failed
#   [build] [therock-eigen configure] -- Performing Test standard_math_library_linked_to_as_m
#   [build] [therock-eigen configure] -- Performing Test standard_math_library_linked_to_as_m - Failed
#   [build] [therock-eigen configure] CMake Error at CMakeLists.txt:108 (message):
#   [build] [therock-eigen configure]   Can't link to the standard math library.  Please report to the Eigen
#   [build] [therock-eigen configure]   developers, telling them about your platform.

therock_cmake_subproject_declare(therock-eigen
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
)
therock_cmake_subproject_provide_package(therock-eigen Eigen3 share/eigen3/cmake)
therock_cmake_subproject_activate(therock-eigen)

add_dependencies(therock-third-party therock-eigen)
