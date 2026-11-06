# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Linux check that the installed thread libraries have CMake names.
# Windows does not build these libraries.

cmake_minimum_required(VERSION 3.25)
project(fftw_threads_client LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)

if(NOT TARGET FFTW3::fftw3_threads)
  message(FATAL_ERROR "FFTW3::fftw3_threads was not defined")
endif()
if(NOT TARGET FFTW3::fftw3f_threads)
  message(FATAL_ERROR "FFTW3::fftw3f_threads was not defined")
endif()

function(_fftw_threads_reject path label)
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

string(TOUPPER "${CMAKE_BUILD_TYPE}" _fftw_cfg)
get_target_property(_fftw_threads FFTW3::fftw3_threads "IMPORTED_LOCATION_${_fftw_cfg}")
get_target_property(_fftwf_threads FFTW3::fftw3f_threads "IMPORTED_LOCATION_${_fftw_cfg}")
_fftw_threads_reject("${_fftw_threads}" "FFTW3::fftw3_threads")
_fftw_threads_reject("${_fftwf_threads}" "FFTW3::fftw3f_threads")

message(STATUS "FFTW3::fftw3_threads: ${_fftw_threads}")
message(STATUS "FFTW3::fftw3f_threads: ${_fftwf_threads}")

add_executable(fftw_threads_client fftw_threads_client.c)
target_link_libraries(fftw_threads_client PRIVATE
  FFTW3::fftw3_threads
  FFTW3::fftw3f_threads
)
