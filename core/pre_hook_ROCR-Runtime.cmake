# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Prefetch LibElf to nullify the shady embedded find modules.
# Maddeningly, the find module calls itself "LibElf" (camel case) but sets
# "LIBELF_FOUND" (uppercase).
if(NOT WIN32)
  find_package(LibElf CONFIG REQUIRED)
  set(LIBELF_FOUND ON)
endif()

# HOST_ASAN passes -Xarch_host -fsanitize=address. ccache drops -Xarch_* unless
# the compile also has -arch, so hsa-runtime never sees a plain -fsanitize=address
# and __has_feature(address_sanitizer) stays false. SANITIZER_AMDGPU is then off,
# GPU pool allocations do not remap the first page to host memory, and the ASan
# chunk header in that page fails the hipFree compare-and-swap. libhsakmt already
# adds this plain flag for HOST_ASAN; the runtime must too.
if(THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address>")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-shared-libsan>")
    add_link_options(-fsanitize=address -shared-libsan)
  endif()
endif()
