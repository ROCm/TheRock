# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

function(therock_add_subdirectory subdir)
  if(NOT THEROCK_CURRENT_SOURCE_DIR)
    set(THEROCK_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  if(NOT THEROCK_CURRENT_BINARY_DIR)
    set(THEROCK_CURRENT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  # Switch on absolute path.
  cmake_path(IS_ABSOLUTE subdir is_abs)
  if(is_abs)
    # Absolute.
    if(ARGC LESS 2)
      message(FATAL_ERROR "therock_add_subdirectory of an absolute path requires a binary dir")
    endif()
    list(GET ARGV 1 binarydir)
    cmake_path(ABSOLUTE_PATH binarydir
      BASE_DIRECTORY "${THEROCK_CURRENT_BINARY_DIR}"
      OUTPUT_VARIABLE THEROCK_CURRENT_BINARY_DIR)
    set(THEROCK_CURRENT_SOURCE_DIR "${subdir}")
  else()
    # Relative
    if(ARGC GREATER 1)
      message(FATAL_ERROR "therock_add_subdirectory with a relative path cannot contain a binary dir")
    endif()
    # Relative
    set(THEROCK_CURRENT_SOURCE_DIR "${THEROCK_CURRENT_SOURCE_DIR}/${subdir}")
    set(THEROCK_CURRENT_BINARY_DIR "${THEROCK_CURRENT_BINARY_DIR}/${subdir}")
  endif()

  # Script we want to include.
  set(script_file "${THEROCK_CURRENT_SOURCE_DIR}/therock.cmake")
  if(NOT EXISTS "${script_file}")
    message(FATAL_ERROR "therock_add_subdirectory(${THEROCK_CURRENT_SOURCE_DIR}): therock.cmake does not exist")
  endif()

  file(MAKE_DIRECTORY "${THEROCK_CURRENT_BINARY_DIR}")
  include("${script_file}")
endfunction()

function(_therock_assert_is_our_directory)
  if(NOT THEROCK_CURRENT_SOURCE_DIR OR NOT THEROCK_CURRENT_BINARY_DIR)
    message(FATAL_ERROR "This function can only be called within a super-project directory added via therock_add_subdirectory()")
  endif()
endfunction()
