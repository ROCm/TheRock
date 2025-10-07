therock_subproject_fetch(therock-yaml-cpp-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/yaml-cpp-0.8.0.tar.gz
  URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
  PATCH_COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/patch_third_party_source.py ${CMAKE_SOURCE_DIR}/patches/third-party/yaml-cpp
)

therock_cmake_subproject_declare(therock-yaml-cpp
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BUILD_DEPS
    therock-fmt
  CMAKE_ARGS
    -DYAML_BUILD_SHARED_LIBS=OFF
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)
therock_cmake_subproject_provide_package(therock-yaml-cpp yaml-cpp lib/cmake)
therock_cmake_subproject_activate(therock-yaml-cpp)

add_dependencies(therock-third-party therock-yaml-cpp)
