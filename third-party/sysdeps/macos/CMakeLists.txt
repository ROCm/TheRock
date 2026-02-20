# macOS system dependencies (alphabetical unless dependency order needed).
# These are built from the common sysdeps directory.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/bzip2" "${CMAKE_CURRENT_BINARY_DIR}/bzip2")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/expat" "${CMAKE_CURRENT_BINARY_DIR}/expat")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/gmp" "${CMAKE_CURRENT_BINARY_DIR}/gmp")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/libbacktrace" "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/liblzma" "${CMAKE_CURRENT_BINARY_DIR}/liblzma")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/ncurses" "${CMAKE_CURRENT_BINARY_DIR}/ncurses")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/sqlite3" "${CMAKE_CURRENT_BINARY_DIR}/sqlite3")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zlib" "${CMAKE_CURRENT_BINARY_DIR}/zlib")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zstd" "${CMAKE_CURRENT_BINARY_DIR}/zstd")

# System deps that depend on the above.
# MPFR depends on GMP.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/mpfr" "${CMAKE_CURRENT_BINARY_DIR}/mpfr")
# Note: hwloc is skipped on macOS because it requires Linux-specific dependencies
# (numactl, libpciaccess). macOS has native APIs for hardware topology.

# Note: The following are Linux-specific and NOT included on macOS:
# - elfutils (uses autotools, Linux-specific ELF format handling)
# - libcap (Linux capabilities)
# - libdrm (Direct Rendering Manager, Linux kernel interface)
# - numactl (NUMA policy library, Linux-specific)
# - libpciaccess (PCI access library, primarily Linux)

# Core sysdeps: This artifact includes libraries that are widely used and are typically
# available on base systems. All of these should be small, fast to build, and high
# fanout to the codebase at large.
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
    therock-libbacktrace
    therock-liblzma
    therock-sqlite3
    therock-zlib
    therock-zstd
)
