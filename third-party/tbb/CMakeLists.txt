therock_subproject_fetch(therock-tbb-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2022.3.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/oneTBB-2022.3.0.tar.gz
  URL_HASH SHA256=01598a46c1162c27253a0de0236f520fd8ee8166e9ebb84a4243574f88e6e50a
)

therock_cmake_subproject_declare(therock-tbb
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DTBB_TEST=OFF
)

therock_cmake_subproject_provide_package(therock-tbb TBB lib/cmake)
therock_cmake_subproject_activate(therock-tbb)

add_dependencies(therock-third-party therock-tbb)
