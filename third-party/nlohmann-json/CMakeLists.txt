therock_subproject_fetch(therock-nlohmann-json-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/json-3.12.0.tar.gz
  URL_HASH SHA256=42F6E95CAD6EC532FD372391373363B62A14AF6D771056DBFC86160E6DFFF7AA
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

if(THEROCK_ENABLE_HIPDNN)
  therock_provide_artifact(nlohmann-json
    DESCRIPTOR artifact-nlohmann-json.toml
    COMPONENTS
      dev
    SUBPROJECT_DEPS
      therock-nlohmann-json
  )
endif()

add_dependencies(therock-third-party therock-nlohmann-json)
