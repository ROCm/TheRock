# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Legacy rocFFT fallback: read FFTW3_INCLUDE_DIRS, then link FFTW3_LIBRARIES.
# Does not link FFTW3::fftw3f and does not look up a threads target.

cmake_minimum_required(VERSION 3.25)
project(fftw_legacy_client LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

if(NOT EXISTS "${FFTW3_INCLUDE_DIRS}/fftw3.h")
  message(FATAL_ERROR "FFTW3_INCLUDE_DIRS has no fftw3.h: ${FFTW3_INCLUDE_DIRS}")
endif()

function(_fftw_legacy_reject path label)
  if(NOT path OR path MATCHES "-NOTFOUND$")
    message(FATAL_ERROR "${label} is empty")
  endif()
  if(FORBIDDEN_BUILD_PREFIX)
    string(FIND "${path}" "${FORBIDDEN_BUILD_PREFIX}" _forbidden_at)
    if(NOT _forbidden_at EQUAL -1)
      message(FATAL_ERROR "${label} still points at the original build: ${path}")
    endif()
  endif()
  if(EXPECTED_PREFIX)
    string(FIND "${path}" "${EXPECTED_PREFIX}/" _expected_at)
    if(NOT _expected_at EQUAL 0)
      message(FATAL_ERROR "${label} is outside the copied prefix: ${path}")
    endif()
  endif()
endfunction()

_fftw_legacy_reject("${FFTW3_INCLUDE_DIRS}" "FFTW3_INCLUDE_DIRS")
_fftw_legacy_reject("${FFTW3_LIBRARY_DIRS}" "FFTW3_LIBRARY_DIRS")
_fftw_legacy_reject("${FFTW3f_INCLUDE_DIRS}" "FFTW3f_INCLUDE_DIRS")
_fftw_legacy_reject("${FFTW3f_LIBRARY_DIRS}" "FFTW3f_LIBRARY_DIRS")

string(TOUPPER "${CMAKE_BUILD_TYPE}" _fftw_cfg)
if(WIN32)
  get_target_property(_fftw3_lib FFTW3::fftw3 "IMPORTED_IMPLIB_${_fftw_cfg}")
else()
  get_target_property(_fftw3_lib FFTW3::fftw3 "IMPORTED_LOCATION_${_fftw_cfg}")
endif()
_fftw_legacy_reject("${_fftw3_lib}" "FFTW3::fftw3")

message(STATUS "FFTW3_INCLUDE_DIRS: ${FFTW3_INCLUDE_DIRS}")
message(STATUS "FFTW3f_INCLUDE_DIRS: ${FFTW3f_INCLUDE_DIRS}")
message(STATUS "FFTW3_LIBRARY_DIRS: ${FFTW3_LIBRARY_DIRS}")
message(STATUS "FFTW3f_LIBRARY_DIRS: ${FFTW3f_LIBRARY_DIRS}")
message(STATUS "FFTW3::fftw3: ${_fftw3_lib}")

add_executable(fftw_legacy_client fftw_client.c)
target_include_directories(fftw_legacy_client PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(fftw_legacy_client PRIVATE ${FFTW3_LIBRARIES})
