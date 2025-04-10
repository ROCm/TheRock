# All system deps (alpha unless if different dependency order needed).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/sqlite3" "${CMAKE_CURRENT_BINARY_DIR}/sqlite3")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zlib" "${CMAKE_CURRENT_BINARY_DIR}/zlib")

therock_provide_artifact(sysdeps
  TARGET_NEUTRAL
  DESCRIPTOR artifact.toml
  COMPONENTS
    dev
    doc
    lib
    run
  SUBPROJECT_DEPS
    therock-sqlite3
)
