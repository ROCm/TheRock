cmake_minimum_required(VERSION 3.25)
project(rocprof-trace-decoder-binary)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  try_run(
    ARCH_DETECT_RUN_RESULT ARCH_DETECT_COMPILE_RESULT
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/compute_release_directory.c
    NO_CACHE
    COMPILE_OUTPUT_VARIABLE ARCH_COMPILE_ERR
    RUN_OUTPUT_VARIABLE ARCH_DETECT_OUTPUT
  )
  if(ARCH_DETECT_COMPILE_RESULT AND ARCH_DETECT_RUN_RESULT EQUAL 0)
    string(STRIP "${ARCH_DETECT_OUTPUT}" RELEASE_ARCH)
    message(STATUS "Detected librocprof-trace-decoder release arch: ${RELEASE_ARCH}")
    set(_shlib_path "${CMAKE_CURRENT_SOURCE_DIR}/binaries/releases/${RELEASE_ARCH}/librocprof-trace-decoder.so")
  else()
    message(
      FATAL_ERROR
      "Failed to find a matching release for librocprof-trace-decoder.so. It may be "
      "unsupported on your architecture and can be disabled with "
      "-DTHEROCK_ENABLE_ROCPROF_TRACE_DECODER_BINARY=OFF. See output of check:\n"
      "${ARCH_COMPILE_ERR}\n"
      "${ARCH_DETECT_OUTPUT}\n"
    )
  endif()

  if(NOT EXISTS "${_shlib_path}")
    message(FATAL_ERROR "Could not find released decoder binary: ${_shlib_path}")
  endif()
  install(FILES "${_shlib_path}" TYPE LIB)
else()
  message(FATAL_ERROR "Unsupported system")
endif()
