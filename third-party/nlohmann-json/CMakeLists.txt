therock_subproject_fetch(therock-nlohmann-json-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/json-3.11.3.tar.gz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)

therock_cmake_subproject_declare(therock-nlohmann-json
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DJSON_BuildTests=OFF
    -DJSON_MultipleHeaders=ON
)
therock_cmake_subproject_provide_package(
  therock-nlohmann-json nlohmann_json share/cmake/nlohmann_json)
therock_cmake_subproject_activate(therock-nlohmann-json)

add_dependencies(therock-third-party therock-nlohmann-json)
