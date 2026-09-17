# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# All system deps (alpha unless if different dependency order needed).
if(THEROCK_ENABLE_SYSDEPS)
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
endif()

# Optional sysdep: gmp
if(THEROCK_ENABLE_SYSDEPS_GMP)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/gmp" "${CMAKE_CURRENT_BINARY_DIR}/gmp")
  therock_provide_artifact(sysdeps-gmp
    TARGET_NEUTRAL
    DESCRIPTOR artifact-gmp.toml
    COMPONENTS
      dev
      doc
      lib
    SUBPROJECT_DEPS
      therock-gmp
  )
endif()

# Optional sysdep: mpfr (depends on gmp)
if(THEROCK_ENABLE_SYSDEPS_MPFR)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/mpfr" "${CMAKE_CURRENT_BINARY_DIR}/mpfr")
  therock_provide_artifact(sysdeps-mpfr
    TARGET_NEUTRAL
    DESCRIPTOR artifact-mpfr.toml
    COMPONENTS
      dev
      doc
      lib
    SUBPROJECT_DEPS
      therock-mpfr
  )
endif()

# Optional sysdep: amd-mesa
if(THEROCK_ENABLE_SYSDEPS_AMD_MESA)
  add_subdirectory(amd-mesa)
  therock_provide_artifact(sysdeps-amd-mesa
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-mesa.toml
    COMPONENTS
      dev
      lib
    SUBPROJECT_DEPS
      therock-amd-mesa
  )
endif()
