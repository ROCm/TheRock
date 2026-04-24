# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Prefetch LibElf to nullify the shady embedded find modules. Homebrew's
# libelf package does not ship a CMake config, so synthesize the target ROCR
# expects when a config package is unavailable.
find_package(LibElf CONFIG QUIET)
if(NOT LibElf_FOUND)
  find_path(LIBELF_INCLUDE_DIRS
    NAMES libelf.h
    HINTS
      /opt/homebrew/opt/libelf
      /usr/local/opt/libelf
      /opt/local
    PATH_SUFFIXES include include/libelf)
  find_library(LIBELF_LIBRARIES
    NAMES elf libelf
    HINTS
      /opt/homebrew/opt/libelf
      /usr/local/opt/libelf
      /opt/local
    PATH_SUFFIXES lib)
  if(LIBELF_INCLUDE_DIRS)
    get_filename_component(_libelf_include_parent "${LIBELF_INCLUDE_DIRS}" DIRECTORY)
    if(EXISTS "${_libelf_include_parent}/libelf/libelf.h")
      list(PREPEND LIBELF_INCLUDE_DIRS "${_libelf_include_parent}")
    endif()
  endif()
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LibElf
    REQUIRED_VARS LIBELF_INCLUDE_DIRS LIBELF_LIBRARIES)
  if(LibElf_FOUND AND NOT TARGET elf::elf)
    add_library(elf::elf UNKNOWN IMPORTED)
    set_target_properties(elf::elf PROPERTIES
      IMPORTED_LOCATION "${LIBELF_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBELF_INCLUDE_DIRS}")
  endif()
elseif(TARGET LibElf::LibElf AND NOT TARGET elf::elf)
  add_library(elf::elf INTERFACE IMPORTED)
  set_target_properties(elf::elf PROPERTIES
    INTERFACE_LINK_LIBRARIES LibElf::LibElf)
endif()

set(LIBELF_INCLUDE_DIR "${LIBELF_INCLUDE_DIRS}")
set(LIBELF_FOUND ON)
