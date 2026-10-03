# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rocprofiler-compute has its own sanitizer resolver, but does not accept
# TheRock's host-only TSAN policy. The generated subproject toolchain has
# already injected the host-scoped compile flags and shared TSAN link options,
# so hide only the unsupported spelling from the redundant project-local
# resolver. Leaving its local sanitizer disabled also prevents its legacy TSAN
# helper from rewriting gfx942/gfx950 targets to :xnack+.
if(THEROCK_SANITIZER STREQUAL "TSAN")
  set(THEROCK_SANITIZER "")
  set(ENABLE_SANITIZER "OFF" CACHE STRING
    "Sanitizer flags are supplied by TheRock's TSAN toolchain" FORCE)
  message(STATUS
    "rocprofiler-compute pre_hook: using TheRock TSAN flags without local GPU target munging")
endif()
