therock_subproject_fetch(therock-libdivide-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/ridiculousfish/libdivide/archive/refs/tags/v5.2.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libdivide-5.2.0.tar.gz
  URL_HASH SHA256=73ae910c4cdbda823b7df2c1e0e1e7427464ebc43fc770b1a30bb598cb703f49
)

therock_cmake_subproject_declare(therock-libdivide
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${THEROCK_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DLIBDIVIDE_BUILD_TESTS=OFF
)
therock_cmake_subproject_provide_package(therock-libdivide libdivide lib/cmake)
therock_cmake_subproject_activate(therock-libdivide)

add_dependencies(therock-third-party therock-libdivide)
