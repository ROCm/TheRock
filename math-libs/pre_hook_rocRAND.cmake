# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(BUILD_SHARED_LIBS ON CACHE BOOL "Workaround https://github.com/ROCm/rocRAND/issues/653." FORCE)

# rocRAND builds device code as C++ through hip::device, so CMAKE_HIP_FLAGS
# (where --offload-compress lives) does not apply. Add it only to those
# targets, after they exist, and leave ordinary C++ targets alone.
if(THEROCK_ASAN_COMPACT)
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  cmake_language(DEFER CALL therock_asan_compact_add_offload_compress "${CMAKE_SOURCE_DIR}")
endif()
