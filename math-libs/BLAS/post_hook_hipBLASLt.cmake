list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_rpath)

therock_set_install_rpath(
  TARGETS
    hipblaslt
  PATHS
    .
)

if(THEROCK_BUILD_TESTING)
  therock_set_install_rpath(
    TARGETS
      hipblaslt-bench
      hipblaslt-bench-extop-amax
      hipblaslt-bench-extop-layernorm
      hipblaslt-bench-extop-matrixtransform
      hipblaslt-bench-extop-softmax
      hipblaslt-bench-groupedgemm-fixed-mk
      hipblaslt-sequence
      hipblaslt-test
    PATHS
      ../lib
      ../lib/host-math/lib
      ../lib/llvm/lib
      ../lib/rocm_sysdeps/lib
  )
endif()
