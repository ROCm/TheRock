# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rocgdb is an autotools-based build wrapped by CMake. Its configure script
# receives CFLAGS/CXXFLAGS and LDFLAGS derived from the CMake cache, so the
# directory compile/link options emitted by therock_sanitizer_configure() never
# reach the nested configure invocation. Mirror both sides here: linker flags
# alone are insufficient because --as-needed drops an otherwise unused TSAN
# runtime from the final ROCgdb executable.
if(THEROCK_SANITIZER STREQUAL "ASAN" OR
   THEROCK_SANITIZER STREQUAL "HOST_ASAN" OR
   THEROCK_SANITIZER STREQUAL "TSAN" OR
   THEROCK_SANITIZER STREQUAL "HOST_TSAN")
  set(_sanitizer_string "address")
  if(THEROCK_SANITIZER STREQUAL "TSAN" OR
     THEROCK_SANITIZER STREQUAL "HOST_TSAN")
    set(_sanitizer_string "thread")
  endif()
  foreach(_var CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    string(APPEND ${_var} " -fsanitize=${_sanitizer_string} -fno-omit-frame-pointer")
  endforeach()
  foreach(_var CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    string(APPEND ${_var} " -fsanitize=${_sanitizer_string} -shared-libsan")
  endforeach()
  message(STATUS "rocgdb pre_hook: appended ${THEROCK_SANITIZER} compile and linker flags")
endif()
