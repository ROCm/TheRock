list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_rpath)

therock_set_install_rpath(
  TARGETS
    rocblas
  PATHS
    .
)

therock_set_install_rpath(
  TARGETS
    rocblas-bench
    rocblas-gemm-tune
    rocblas-test
  PATHS
    ../lib
    ../lib/host-math/lib
    ../lib/llvm/lib
    ../lib/rocm_sysdeps/lib
)
