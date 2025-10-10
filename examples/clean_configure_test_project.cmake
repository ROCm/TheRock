if(EXISTS "${BINARY_DIR}")
  message(STATUS "Removing build dir ${BINARY_DIR}")
  file(REMOVE_RECURSE "${BINARY_DIR}")
endif()

set(propagate_vars
  CMAKE_CXX_COMPILER
  THEROCK_ENABLE_BLAS
  THEROCK_ENABLE_FFT
  THEROCK_ENABLE_HIP
  THEROCK_ENABLE_HIPDNN
  THEROCK_ENABLE_MIOPEN
  THEROCK_ENABLE_PRIM
  THEROCK_ENABLE_RAND
  THEROCK_ENABLE_RCCL
  THEROCK_ENABLE_SOLVER
  THEROCK_ENABLE_SPARSE
  CMAKE_HIP_PLATFORM
  CMAKE_HIP_COMPILER
  CMAKE_HIP_COMPILER_ROCM_ROOT
)

set(build_options)
foreach(var_name ${propagate_vars})
  if(DEFINED ${var_name})
    list(APPEND build_options "-D${var_name}=${${var_name}}")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}" --build-and-test
    "${SOURCE_DIR}"
    "${BINARY_DIR}"
    --build-generator "${GENERATOR}"
    --build-options ${build_options}
    --test-command "${CMAKE_CTEST_COMMAND}" --output-on-failure --verbose
  RESULT_VARIABLE CMD_RESULT
)

if(CMD_RESULT)
  message(FATAL_ERROR "Failed to execute test process")
endif()
