therock_subproject_fetch(therock-spdlog-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/spdlog-1.15.3.tar.gz
  URL_HASH SHA256=15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67
)

therock_cmake_subproject_declare(therock-spdlog
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BUILD_DEPS
    therock-fmt
  CMAKE_ARGS
    -DSPDLOG_BUILD_EXAMPLE=OFF
    -DSPDLOG_INSTALL=ON
    -DSPDLOG_FMT_EXTERNAL=ON
)
therock_cmake_subproject_provide_package(therock-spdlog spdlog lib/cmake)
therock_cmake_subproject_activate(therock-spdlog)

therock_provide_artifact(spdlog
  DESCRIPTOR artifact-spdlog.toml
  COMPONENTS
    dev
  SUBPROJECT_DEPS
    therock-spdlog
)

add_dependencies(therock-third-party therock-spdlog)
