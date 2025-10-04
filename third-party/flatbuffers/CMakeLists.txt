therock_subproject_fetch(therock-flatbuffers-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/google/flatbuffers/archive/refs/tags/v25.9.23.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/flatbuffers-25.9.23.tar.gz
  URL_HASH SHA256=9102253214dea6ae10c2ac966ea1ed2155d22202390b532d1dea64935c518ada
)

therock_cmake_subproject_declare(therock-flatbuffers
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BUILD_DEPS
  CMAKE_ARGS
)
therock_cmake_subproject_provide_package(therock-flatbuffers flatbuffers lib/cmake)
therock_cmake_subproject_activate(therock-flatbuffers)

add_dependencies(therock-third-party therock-flatbuffers)
