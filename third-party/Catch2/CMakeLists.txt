therock_subproject_fetch(therock-catch2-sources
  CMAKE_PROJECT
  URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.0.tar.gz
  URL_HASH SHA256=5b10cd536fa3818112a82820ce0787bd9f2a906c618429e7c4dea639983c8e88
)

therock_cmake_subproject_declare(therock-catch2
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
)
therock_cmake_subproject_provide_package(therock-catch2 Catch2 lib/cmake)
therock_cmake_subproject_activate(therock-catch2)

add_dependencies(therock-third-party therock-catch2)
