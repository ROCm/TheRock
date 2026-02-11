therock_subproject_fetch(therock-msgpack-cxx-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/msgpack/msgpack-c/releases/download/cpp-7.0.0/msgpack-cxx-7.0.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/msgpack-cxx-7.0.0.tar.gz
  URL_HASH SHA256=7504b7af7e7b9002ce529d4f941e1b7fb1fb435768780ce7da4abaac79bb156f
)

therock_cmake_subproject_declare(therock-msgpack-cxx
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DMSGPACK_USE_BOOST=OFF
)
therock_cmake_subproject_provide_package(
  therock-msgpack-cxx msgpack-cxx lib/cmake/msgpack-cxx)
therock_cmake_subproject_activate(therock-msgpack-cxx)

add_dependencies(therock-third-party therock-msgpack-cxx)
