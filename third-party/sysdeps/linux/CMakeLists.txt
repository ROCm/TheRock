# All system deps (alpha unless if different dependency order needed).
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/bzip2" "${CMAKE_CURRENT_BINARY_DIR}/bzip2")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/expat" "${CMAKE_CURRENT_BINARY_DIR}/expat")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/gmp" "${CMAKE_CURRENT_BINARY_DIR}/gmp")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/libbacktrace" "${CMAKE_CURRENT_BINARY_DIR}/libbacktrace")
add_subdirectory(libcap)
add_subdirectory(libdrm)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/liblzma" "${CMAKE_CURRENT_BINARY_DIR}/liblzma")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/ncurses" "${CMAKE_CURRENT_BINARY_DIR}/ncurses")
add_subdirectory(numactl)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/sqlite3" "${CMAKE_CURRENT_BINARY_DIR}/sqlite3")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zlib" "${CMAKE_CURRENT_BINARY_DIR}/zlib")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/zstd" "${CMAKE_CURRENT_BINARY_DIR}/zstd")

# System deps that depend on the above.
add_subdirectory(elfutils)
# MPFR depends on GMP.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/mpfr" "${CMAKE_CURRENT_BINARY_DIR}/mpfr")
# libpciaccess for PCI device access.
add_subdirectory(libpciaccess)
# hwloc depends on numactl and libpciaccess.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/hwloc" "${CMAKE_CURRENT_BINARY_DIR}/hwloc")

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
    therock-elfutils
    therock-libbacktrace
    therock-libcap
    therock-libdrm
    therock-liblzma
    therock-numactl
    therock-sqlite3
    therock-zlib
    therock-zstd
)

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

# Optional sysdep: expat
if(THEROCK_ENABLE_SYSDEPS_EXPAT)
  therock_provide_artifact(sysdeps-expat
    TARGET_NEUTRAL
    DESCRIPTOR artifact-expat.toml
    COMPONENTS
      dev
      doc
      lib
    SUBPROJECT_DEPS
      therock-expat
  )
endif()

# Optional sysdep: gmp
if(THEROCK_ENABLE_SYSDEPS_GMP)
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

# Optional sysdep: ncurses
if(THEROCK_ENABLE_SYSDEPS_NCURSES)
  therock_provide_artifact(sysdeps-ncurses
    TARGET_NEUTRAL
    DESCRIPTOR artifact-ncurses.toml
    COMPONENTS
      dev
      doc
      lib
    SUBPROJECT_DEPS
      therock-ncurses
  )
endif()

# Optional sysdep: libpciaccess
if(THEROCK_ENABLE_SYSDEPS_LIBPCIACCESS)
  therock_provide_artifact(sysdeps-libpciaccess
    TARGET_NEUTRAL
    DESCRIPTOR artifact-libpciaccess.toml
    COMPONENTS
      dev
      lib
    SUBPROJECT_DEPS
      therock-libpciaccess
  )
endif()

# Optional sysdep: hwloc (depends on numactl and libpciaccess)
if(THEROCK_ENABLE_SYSDEPS_HWLOC)
  therock_provide_artifact(sysdeps-hwloc
    TARGET_NEUTRAL
    DESCRIPTOR artifact-hwloc.toml
    COMPONENTS
      dev
      doc
      lib
    SUBPROJECT_DEPS
      therock-hwloc
  )
endif()
