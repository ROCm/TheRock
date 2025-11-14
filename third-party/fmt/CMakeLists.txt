therock_subproject_fetch(therock-fmt-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/fmtlib/fmt/releases/download/11.1.3/fmt-11.1.3.zip
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fmt-11.1.3.zip
  URL_HASH SHA256=7df2fd3426b18d552840c071c977dc891efe274051d2e7c47e2c83c3918ba6df
)

therock_cmake_subproject_declare(therock-fmt
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DFMT_DOC=OFF
    -DFMT_TEST=OFF
)
therock_cmake_subproject_provide_package(
  therock-fmt fmt lib/cmake/fmt)
therock_cmake_subproject_activate(therock-fmt)

if(THEROCK_ENABLE_HIPDNN)
  therock_provide_artifact(fmt
    TARGET_NEUTRAL
    DESCRIPTOR artifact-fmt.toml
    COMPONENTS
      dev
    SUBPROJECT_DEPS
      therock-fmt
  )
endif()

add_dependencies(therock-third-party therock-fmt)
