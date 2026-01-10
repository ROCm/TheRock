therock_subproject_fetch(therock-catch2-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/catchorg/Catch2/archive/refs/tags/v3.8.1.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/Catch2-3.8.1.tar.gz
  URL_HASH SHA256=18b3f70ac80fccc340d8c6ff0f339b2ae64944782f8d2fca2bd705cf47cadb79
)

therock_cmake_subproject_declare(therock-catch2
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
)
therock_cmake_subproject_provide_package(therock-catch2 Catch2 lib/cmake)
therock_cmake_subproject_activate(therock-catch2)

add_dependencies(therock-third-party therock-catch2)
