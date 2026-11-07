# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# HOST_ASAN passes -Xarch_host -fsanitize=address. ccache drops -Xarch_* unless
# the compile also has -arch, so hsa-runtime never sees a plain -fsanitize=address
# and __has_feature(address_sanitizer) stays false. SANITIZER_AMDGPU is then off,
# GPU pool allocations do not remap the first page to host memory, and the ASan
# chunk header in that page fails the hipFree compare-and-swap. libhsakmt already
# adds this plain flag for HOST_ASAN; the runtime must too.
#
# The old LibElf prefetch that lived in this file was removed in #9033. The hook
# is back because therock_cmake_subproject_declare includes
# pre_hook_<target>.cmake when that file exists.
if(THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address>")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-shared-libsan>")
    add_link_options(-fsanitize=address -shared-libsan)
  endif()
endif()
