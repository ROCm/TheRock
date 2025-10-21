# TODO: HIP language support not based on an /opt/rocm installed SDK
# must have these three cache variables set to avoid language setup
# error. https://github.com/ROCm/TheRock/issues/102
set(HIP_CMAKE_ARGS
  -DCMAKE_HIP_PLATFORM=amd
  "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${THEROCK_BINARY_DIR}/dist/rocm"
)

if(WIN32)
  # On Windows, only a unified compiler is supported (i.e. use the HIP compiler
  # for everything).
  list(APPEND HIP_CMAKE_ARGS
    "-DCMAKE_CXX_COMPILER=${THEROCK_BINARY_DIR}/dist/rocm/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  )
else()
  # On everything else, the HIP language compiler can be independent from the
  # overall CXX compiler.
  list(APPEND HIP_CMAKE_ARGS
    "-DCMAKE_HIP_COMPILER=${THEROCK_BINARY_DIR}/dist/rocm/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}"
  )
endif()

add_test(
  NAME therock-examples-cpp-sdk-user
  COMMAND
    "${CMAKE_COMMAND}" -E env "CMAKE_PREFIX_PATH=${THEROCK_BINARY_DIR}/dist/rocm" --
    "${CMAKE_COMMAND}"
        "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cpp-sdk-user"
        "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cpp-sdk-user"
        "-DGENERATOR=${CMAKE_GENERATOR}"
        "-DTHEROCK_ENABLE_BLAS=${THEROCK_ENABLE_BLAS}"
        "-DTHEROCK_ENABLE_FFT=${THEROCK_ENABLE_FFT}"
        "-DTHEROCK_ENABLE_HIP=${THEROCK_ENABLE_HIP_RUNTIME}"
        "-DTHEROCK_ENABLE_HIPDNN=${THEROCK_ENABLE_HIPDNN}"
        "-DTHEROCK_ENABLE_MIOPEN=${THEROCK_ENABLE_MIOPEN}"
        "-DTHEROCK_ENABLE_PRIM=${THEROCK_ENABLE_PRIM}"
        "-DTHEROCK_ENABLE_RAND=${THEROCK_ENABLE_RAND}"
        "-DTHEROCK_ENABLE_RCCL=${THEROCK_ENABLE_RCCL}"
        "-DTHEROCK_ENABLE_SOLVER=${THEROCK_ENABLE_SOLVER}"
        "-DTHEROCK_ENABLE_SPARSE=${THEROCK_ENABLE_SPARSE}"
        ${HIP_CMAKE_ARGS}
      -P "${CMAKE_CURRENT_SOURCE_DIR}/clean_configure_test_project.cmake"
)
