# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Built here, shipped in the test component. Declared by profiler/CMakeLists.txt.

cmake_minimum_required(VERSION 3.25)
project(profiler-hub-install-test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT PROFILER_HUB_ARTIFACT_PREFIX)
  message(
    FATAL_ERROR
    "PROFILER_HUB_ARTIFACT_PREFIX must name an assembled profiler-hub artifact tree"
  )
endif()

function(add_profiler_hub_install_test)
  cmake_parse_arguments(ARG "" "PACKAGE;TARGET" "HEADER" ${ARGN})

  if(NOT ARG_PACKAGE OR NOT ARG_TARGET OR NOT ARG_HEADER)
    message(
      FATAL_ERROR
      "add_profiler_hub_install_test requires PACKAGE, TARGET, and at least one HEADER"
    )
  endif()

  find_package(${ARG_PACKAGE} CONFIG REQUIRED
    PATHS "${PROFILER_HUB_ARTIFACT_PREFIX}"
    NO_DEFAULT_PATH
  )
  message(STATUS "${ARG_PACKAGE} found in ${PROFILER_HUB_ARTIFACT_PREFIX}")

  # HEADER is an ordered list of public include paths. The first candidate
  # present in the installed package is the one the test compiles against.
  set(include_dirs "")
  foreach(include_var IN ITEMS "${ARG_PACKAGE}_INCLUDE_DIR" "${ARG_PACKAGE}_INCLUDE_DIRS")
    if(DEFINED ${include_var})
      list(APPEND include_dirs "${${include_var}}")
    endif()
  endforeach()
  if(TARGET ${ARG_TARGET})
    get_target_property(iface_includes ${ARG_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    if(iface_includes)
      foreach(inc IN LISTS iface_includes)
        if(NOT inc MATCHES "\\$<")
          list(APPEND include_dirs "${inc}")
        endif()
      endforeach()
    endif()
  endif()
  list(APPEND include_dirs "${PROFILER_HUB_ARTIFACT_PREFIX}/include")
  list(REMOVE_DUPLICATES include_dirs)

  set(header "")
  foreach(candidate IN LISTS ARG_HEADER)
    foreach(inc IN LISTS include_dirs)
      if(EXISTS "${inc}/${candidate}")
        set(header "${candidate}")
        break()
      endif()
    endforeach()
    if(header)
      break()
    endif()
  endforeach()

  if(NOT header)
    message(
      FATAL_ERROR
      "${ARG_PACKAGE}: none of the candidate headers were found "
      "(${ARG_HEADER}) under: ${include_dirs}"
    )
  endif()
  message(STATUS "${ARG_PACKAGE}: install test will include <${header}>")

  set(test_name "test_${ARG_PACKAGE}")
  set(pkg_name "${ARG_PACKAGE}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/test_template.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp"
    @ONLY
  )
  add_executable(${test_name} "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cpp")
  target_link_libraries(${test_name} PRIVATE ${ARG_TARGET})
  install(TARGETS ${test_name} RUNTIME DESTINATION bin)
endfunction()

# storage.hpp is moving from profiler-hub/ to profiler-hub/cpp/. Accept either
# location until the old path is removed.
add_profiler_hub_install_test(
  PACKAGE profiler-hub
  TARGET profiler-hub::profiler-hub
  HEADER
    profiler-hub/cpp/storage.hpp
    profiler-hub/storage.hpp
)
