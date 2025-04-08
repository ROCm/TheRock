cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(
  roctx64-compat
  LANGUAGES C CXX
  VERSION 4.1.0  # Last roctracer release
)

find_package(rocprofiler-sdk-roctx REQUIRED)

install(DIRECTORY include/roctracer TYPE INCLUDE)
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/roctracer-compat-config.cmake"
  DESTINATION lib/cmake/roctracer-compat
)
