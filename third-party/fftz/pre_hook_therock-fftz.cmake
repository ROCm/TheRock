# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(WIN32)
  # FFTZ uses these platform macros, which CMake's MSVC frontend supplies
  # by default but its GNU Clang frontend does not.
  add_compile_definitions(WIN32 _WINDOWS)
endif()
