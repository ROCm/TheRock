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
  if(WIN32)
    # This helper is Linux only. In the future, we can have separate DLL_NAMES
    # and verify.
    return()
  endif()

  if(NOT IS_ABSOLUTE ARG_PATH)
    cmake_path(ABSOLUTE_PATH ARG_PATH BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  separate_arguments(CMAKE_C_COMPILER_LIST UNIX_COMMAND "${CMAKE_C_COMPILER}")

  set(CLANG_EXECUTABLE
    "${CMAKE_SOURCE_DIR}/build/lib/llvm/bin/clang"
    CACHE FILEPATH "Clang executable"
  )

  execute_process(
    COMMAND ${CLANG_EXECUTABLE} --print-file-name=libclang_rt.asan-x86_64.so
    OUTPUT_VARIABLE ASAN_RUNTIME_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(ASAN_PRELOAD "${ASAN_RUNTIME_PATH}")

  foreach(lib_name ${ARG_LIB_NAMES})
    add_test(
      NAME therock-validate-shared-lib-${lib_name}
      COMMAND
        env
          LD_PRELOAD=${ASAN_PRELOAD}
          ${THEROCK_SANITIZER_LAUNCHER}
          "${Python3_EXECUTABLE}" "${THEROCK_SOURCE_DIR}/build_tools/validate_shared_library.py"
          "${ARG_PATH}/${lib_name}"
    )
  endforeach()
endfunction()