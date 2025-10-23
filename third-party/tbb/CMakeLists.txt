therock_subproject_fetch(therock-tbb-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/uxlfoundation/oneTBB/archive/refs/tags/2018_U6.tar.gz
  # TODO: Update this to be the ROCm S3 bucket URL instead once uploaded
  URL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/2018_U6.tar.gz
  URL_HASH SHA256=0ebb5fc877871ef15f7395d6e3c86de4ffedb820dc336383a3ab71fc39426aa7
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