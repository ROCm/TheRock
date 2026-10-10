# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TheRock supplies host TSAN flags. Disable the redundant local sanitizer
# resolver so it cannot rewrite GPU targets to :xnack+.
if(THEROCK_SANITIZER STREQUAL "TSAN")
  set(THEROCK_SANITIZER "")
  set(ENABLE_SANITIZER OFF CACHE STRING "Sanitizer flags supplied by TheRock" FORCE)
endif()
