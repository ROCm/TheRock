list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_rpath)

therock_set_install_rpath(
  TARGETS
    hipblas
  PATHS
    .
)

therock_set_install_rpath(
  TARGETS
    hipblas-bench
    hipblas-test
    hipblas_v2-bench
    hipblas_v2-test
  PATHS
    ../lib
    ../lib/host-math/lib
    ../lib/llvm/lib
    ../lib/rocm_sysdeps/lib
)
