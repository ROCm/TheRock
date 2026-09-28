# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Adds a test for shared libraries under a common path.
# PATH: Common path (relative to CMAKE_CURRENT_BINARY_DIR if not absolute)
# LIB_NAMES: Library names to validate
# FORBID_NEEDED: optional regex; forwarded as --forbid-needed (Linux ELF DT_NEEDED)
# FORBID_DYNSYM: optional regex; forwarded as --forbid-dynsym (Linux defined dynsym)
function(therock_test_validate_shared_lib)
  cmake_parse_arguments(
    PARSE_ARGV 0 ARG
    ""
    "PATH;FORBID_NEEDED;FORBID_DYNSYM"
    "LIB_NAMES"
  )

  # Skip shared-library dlopen validation for sanitizer builds.
  if(NOT "${THEROCK_SANITIZER}" STREQUAL "")
    return()
  endif()

  if(WIN32)
    # This helper is Linux only. In the future, we can have separate DLL_NAMES
    # and verify.
    return()
  endif()

  # validate_shared_library.py exits 2 if --forbid-needed / --forbid-dynsym are
  # passed anywhere other than Linux (ELF-only gates). Registering the test on
  # another POSIX host would produce a test that can never pass.
  if(ARG_FORBID_NEEDED OR ARG_FORBID_DYNSYM)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
      return()
    endif()
  endif()

  if(NOT IS_ABSOLUTE ARG_PATH)
    cmake_path(ABSOLUTE_PATH ARG_PATH BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  foreach(lib_name ${ARG_LIB_NAMES})
    set(_validate_cmd
      "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/validate_shared_library.py"
        "${ARG_PATH}/${lib_name}"
    )
    if(ARG_FORBID_NEEDED)
      list(APPEND _validate_cmd --forbid-needed "${ARG_FORBID_NEEDED}")
    endif()
    if(ARG_FORBID_DYNSYM)
      list(APPEND _validate_cmd --forbid-dynsym "${ARG_FORBID_DYNSYM}")
    endif()
    add_test(
      NAME therock-validate-shared-lib-${lib_name}
      COMMAND ${_validate_cmd}
    )
  endforeach()
endfunction()
