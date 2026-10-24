# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Included by the CMAKE_PROJECT_INCLUDE file that therock_subproject.cmake
# generates for hipBLASLt while its coverage is on. hipBLASLt's coverage-only
# test link names a bare `rocroller`, which its own build resolves by having
# rocRoller in-tree. Under TheRock, rocRoller is a separate subproject imported
# as roc::rocroller, so the bare name reaches the linker as -lrocroller with no
# -L to find it. Once the top-level CMakeLists.txt has run, and with it the
# find_package that imports roc::rocroller, the name gets a target that
# forwards to the imported one.

get_property(_therock_coverage_rocroller_deferred GLOBAL
  PROPERTY THEROCK_COVERAGE_ROCROLLER_DEFERRED)
if(NOT _therock_coverage_rocroller_deferred)
  set_property(GLOBAL PROPERTY THEROCK_COVERAGE_ROCROLLER_DEFERRED TRUE)
  function(_therock_coverage_rocroller_target)
    if(TARGET roc::rocroller AND NOT TARGET rocroller)
      add_library(rocroller INTERFACE)
      target_link_libraries(rocroller INTERFACE roc::rocroller)
    endif()
  endfunction()
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL _therock_coverage_rocroller_target)
endif()
