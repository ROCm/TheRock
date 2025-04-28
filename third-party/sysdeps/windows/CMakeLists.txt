# All system deps (alpha unless if different dependency order needed).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/bzip2" "${CMAKE_CURRENT_BINARY_DIR}/bzip2")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/sqlite3" "${CMAKE_CURRENT_BINARY_DIR}/sqlite3")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zlib" "${CMAKE_CURRENT_BINARY_DIR}/zlib")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zstd" "${CMAKE_CURRENT_BINARY_DIR}/zstd")

therock_provide_artifact(sysdeps
  TARGET_NEUTRAL
  DESCRIPTOR artifact.toml
  COMPONENTS
    dev
    lib
  SUBPROJECT_DEPS
    therock-bzip2
    therock-sqlite3
    therock-zlib
    therock-zstd
)
