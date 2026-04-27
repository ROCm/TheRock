# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# On Linux/macOS, prefetch LibElf from bundled elfutils to nullify the shady
# embedded find modules. On Windows, we pre-set LIBELF_FOUND and paths so
# FindLibElf.cmake is skipped (it has add_subdirectory issues in the subproject
# system). The oclelf library is built via add_subdirectory in hsa-runtime's
# CMakeLists.txt.
if(WIN32)
  # Set up paths for AMD's bundled elftoolchain libelf
  if(AMD_LIBELF_PATH)
    set(_elftoolchain_root "${AMD_LIBELF_PATH}/..")
    set(LIBELF_INCLUDE_DIR
      "${AMD_LIBELF_PATH}"
      "${_elftoolchain_root}/common/win32"
      "${_elftoolchain_root}/common"
      CACHE STRING "LibElf include directories" FORCE)
    set(LIBELF_FOUND ON CACHE BOOL "LibElf found" FORCE)
    set(LibElf_FOUND ON CACHE BOOL "LibElf found" FORCE)
    set(USE_AMD_LIBELF "yes" CACHE STRING "" FORCE)
  endif()
else()
  # Maddeningly, the find module calls itself "LibElf" (camel case) but sets
  # "LIBELF_FOUND" (uppercase).
  find_package(LibElf CONFIG REQUIRED)
  set(LIBELF_FOUND ON)
endif()
