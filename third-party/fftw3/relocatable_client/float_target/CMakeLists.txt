# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Link FFTW3::fftw3f and take the header from that target's include directory.
# No extra include path is added. The float prefix is used on its own.

cmake_minimum_required(VERSION 3.25)
project(fftw_float_target_client LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(FFTW3f CONFIG REQUIRED)

if(NOT EXISTS "${FFTW3f_INCLUDE_DIRS}/fftw3.h")
  message(FATAL_ERROR "FFTW3f_INCLUDE_DIRS has no fftw3.h: ${FFTW3f_INCLUDE_DIRS}")
endif()

function(_fftw_float_reject path label)
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

_fftw_float_reject("${FFTW3f_INCLUDE_DIRS}" "FFTW3f_INCLUDE_DIRS")
get_target_property(_fftw_float_includes FFTW3::fftw3f INTERFACE_INCLUDE_DIRECTORIES)
_fftw_float_reject("${_fftw_float_includes}" "FFTW3::fftw3f include")
if(NOT EXISTS "${_fftw_float_includes}/fftw3.h")
  message(FATAL_ERROR "FFTW3::fftw3f include has no fftw3.h: ${_fftw_float_includes}")
endif()

message(STATUS "FFTW3f_INCLUDE_DIRS: ${FFTW3f_INCLUDE_DIRS}")
message(STATUS "FFTW3::fftw3f include: ${_fftw_float_includes}")

add_executable(fftw_float_client fftw_float_client.c)
target_link_libraries(fftw_float_client PRIVATE FFTW3::fftw3f)
