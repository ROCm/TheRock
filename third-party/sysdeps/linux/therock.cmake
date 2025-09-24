# All system deps (alpha unless if different dependency order needed).
therock_add_subdirectory("${THEROCK_CURRENT_SOURCE_DIR}/../common/bzip2" "${THEROCK_CURRENT_BINARY_DIR}/bzip2")
therock_add_subdirectory(libdrm)
therock_add_subdirectory(numactl)
therock_add_subdirectory("${THEROCK_CURRENT_SOURCE_DIR}/../common/sqlite3" "${THEROCK_CURRENT_BINARY_DIR}/sqlite3")
therock_add_subdirectory("${THEROCK_CURRENT_SOURCE_DIR}/../common/zlib" "${THEROCK_CURRENT_BINARY_DIR}/zlib")
therock_add_subdirectory("${THEROCK_CURRENT_SOURCE_DIR}/../common/zstd" "${THEROCK_CURRENT_BINARY_DIR}/zstd")

# System deps that depend on the above.
therock_add_subdirectory(elfutils)

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
