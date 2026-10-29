# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rocgdb is an autotools-based build wrapped by CMake. Its configure script
# receives LDFLAGS derived from CMAKE_SHARED_LINKER_FLAGS, so add_link_options()
# from therock_sanitizer_configure() never reaches the link line. Mirror those
# flags into CMAKE_*_LINKER_FLAGS here so sanitizer runs work end-to-end.
if(THEROCK_SANITIZER STREQUAL "ASAN" OR
   THEROCK_SANITIZER STREQUAL "HOST_ASAN" OR
   THEROCK_SANITIZER STREQUAL "TSAN")
  set(_sanitizer_string "address")
  if(THEROCK_SANITIZER STREQUAL "TSAN")
    set(_sanitizer_string "thread")
  endif()
  foreach(_var CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    string(APPEND ${_var} " -fsanitize=${_sanitizer_string} -shared-libsan")
  endforeach()
  message(STATUS "rocgdb pre_hook: appended ${THEROCK_SANITIZER} flags to CMAKE_{EXE,SHARED}_LINKER_FLAGS")
endif()
