# Adds a test for shared libraries under a common path.
# PATH: Common path (relative to CMAKE_CURRENT_BINARY_DIR if not absolute)
# LIB_NAMES: Library names to validate
function(therock_test_validate_shared_lib)
  cmake_parse_arguments(
    PARSE_ARGV 0 ARG
    ""
    "PATH"
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

  if(NOT IS_ABSOLUTE ARG_PATH)
    cmake_path(ABSOLUTE_PATH ARG_PATH BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  foreach(lib_name ${ARG_LIB_NAMES})
    add_test(
      NAME therock-validate-shared-lib-${lib_name}
      COMMAND
        "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/validate_shared_library.py"
          "${ARG_PATH}/${lib_name}"
    )
  endforeach()
endfunction()
