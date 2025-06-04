cmake_minimum_required(VERSION 3.25)
project(rocprof-trace-decoder-binary)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_shlib_path "${CMAKE_CURRENT_BINARY_DIR}/../download/opt/rocm/lib/librocprof-trace-decoder.so")
  install(FILES "${_shlib_path}" TYPE LIB)
else()
  message(FATAL_ERROR "Unsupported system")
endif()
