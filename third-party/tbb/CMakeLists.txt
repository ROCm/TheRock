therock_subproject_fetch(therock-tbb-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/ajanicijamd/oneTBB/archive/refs/tags/v2018.6.01.tar.gz
  # TODO: Update this to be the ROCm S3 bucket URL instead once uploaded
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/oneTBB-2018.6.01.tar.gz
  URL_HASH SHA256=7769d82c48dde8fb250b1928891ddd70e16ceec32f71e69c7d97681e012aa6be
)

therock_cmake_subproject_declare(therock-tbb
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
)

therock_cmake_subproject_provide_package(therock-tbb tbb lib/cmake)
therock_cmake_subproject_activate(therock-tbb)

add_dependencies(therock-third-party therock-tbb)
