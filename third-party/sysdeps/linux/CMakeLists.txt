# All system deps (alpha unless if different dependency order needed).
add_subdirectory(bzip2)
add_subdirectory(libdrm)
add_subdirectory(numactl)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/sqlite3" "${CMAKE_CURRENT_BINARY_DIR}/sqlite3")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zlib" "${CMAKE_CURRENT_BINARY_DIR}/zlib")
add_subdirectory(zstd)

# System deps that depend on the above.
add_subdirectory(elfutils)

therock_provide_artifact(sysdeps
  TARGET_NEUTRAL
  DESCRIPTOR artifact.toml
  COMPONENTS
    dev
    doc
    lib
    run
  SUBPROJECT_DEPS
    therock-bzip2
    therock-elfutils
    therock-libdrm
    therock-numactl
    therock-sqlite3
    therock-zlib
    therock-zstd
)
