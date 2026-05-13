# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message(STATUS "Customizing zlib options for TheRock")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# zlib 1.3.2 forces CMAKE_DEBUG_POSTFIX="d" on MSVC, producing zsd.lib for
# Debug builds. TheRock's zlib-config.cmake hardcodes zs.lib, so strip the
# postfix so a Debug super-project still finds the static archive.
if(MSVC)
  set(CMAKE_DEBUG_POSTFIX "")
endif()
