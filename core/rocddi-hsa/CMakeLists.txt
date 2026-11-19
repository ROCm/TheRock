# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(therock-rocddi-hsa LANGUAGES C)

set(ROCM_RUNTIMES_BINARY_DIR "" CACHE PATH "Upstream runtime build tree")
set(ROCM_RUNTIMES_SOURCE_DIR "" CACHE PATH "Upstream runtime sources")
set(_prefix "${ROCM_RUNTIMES_BINARY_DIR}")
file(GLOB _images "${_prefix}/lib/libhsa-runtime64.so.1.*.*")
list(LENGTH _images _count)
if(NOT _count EQUAL 1 OR NOT EXISTS "${ROCM_RUNTIMES_SOURCE_DIR}/api-headers/include/hsa/hsa.h")
  message(FATAL_ERROR "Expected one built HSA image and its producer headers")
endif()
list(GET _images 0 _image)
cmake_path(GET _image FILENAME _name)
if(NOT _name MATCHES "^libhsa-runtime64\\.so\\.(1\\.[0-9]+\\.[0-9]+)$")
  message(FATAL_ERROR "Unexpected HSA image filename")
endif()
set(HSA_VERSION "${CMAKE_MATCH_1}")
set(HSA_SONAME libhsa-runtime64.so.1)
find_program(ROCM_RUNTIMES_READELF NAMES llvm-readelf readelf REQUIRED)
execute_process(COMMAND "${ROCM_RUNTIMES_READELF}" --dynamic "${_image}"
  OUTPUT_VARIABLE _dynamic COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCH "\\(SONAME\\)[^\n]*\\[([^]]+)\\]" _entry "${_dynamic}")
if(NOT CMAKE_MATCH_1 STREQUAL HSA_SONAME)
  message(FATAL_ERROR "Expected HSA SONAME ${HSA_SONAME}")
endif()
foreach(_link libhsa-runtime64.so libhsa-runtime64.so.1)
  if(NOT EXISTS "${_prefix}/lib/${_link}")
    message(FATAL_ERROR "Missing producer library link ${_link}")
  endif()
endforeach()

# Promote the producer's image and matching headers into the SDK layout.
install(DIRECTORY "${_prefix}/lib/" DESTINATION lib
  FILES_MATCHING PATTERN "libhsa-runtime64.so*")
include(PublicHsaHeaders.cmake)
rocm_runtimes_prepare_hsa_headers(
  "${ROCM_RUNTIMES_SOURCE_DIR}/api-headers/include/hsa" "${CMAKE_CURRENT_BINARY_DIR}/public-hsa")
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/public-hsa/" DESTINATION include/hsa)
install(FILES LICENSE DESTINATION share/doc/rocddi-hsa)
include(CMakePackageConfigHelpers)
configure_package_config_file(hsa-runtime64-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/hsa-runtime64-config.cmake"
  INSTALL_DESTINATION lib/cmake/hsa-runtime64)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/hsa-runtime64-config-version.cmake"
  VERSION "${HSA_VERSION}" COMPATIBILITY SameMajorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/hsa-runtime64-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/hsa-runtime64-config-version.cmake"
  DESTINATION lib/cmake/hsa-runtime64)
