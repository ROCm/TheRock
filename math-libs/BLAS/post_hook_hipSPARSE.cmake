list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_rpath)

therock_set_install_rpath(
  TARGETS
    hipsparse
  PATHS
    .
)

therock_set_install_rpath(
  TARGETS
    hipsparse-bench
    hipsparse-test
  PATHS
    ../lib
    ../lib/llvm/lib
    ../lib/rocm_sysdeps/lib
)
