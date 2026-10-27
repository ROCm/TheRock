# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# DASan's fast allocation-stack unwinder traverses the HIP runtime between the
# HSA allocation interceptor and the application. Preserve those frames in the
# opt-in DASan build so allocation reports can reach the application's call site.
if(THEROCK_ENABLE_DASAN AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_options(-fno-omit-frame-pointer)
endif()
