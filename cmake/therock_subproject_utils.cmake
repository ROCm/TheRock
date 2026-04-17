# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Recursively gets all build system targets defined in a directory and all of
# its subdirectories.
function(therock_get_all_targets var dir)
  get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  list(APPEND ${var} ${targets})
  get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(subdir ${subdirs})
    therock_get_all_targets("${var}" "${subdir}")
  endforeach()
  set(${var} "${${var}}" PARENT_SCOPE)
endfunction()


# RPATH padding for patchelf in-place overwrite.
#
# Some older kernels (notably RHEL 8.10 / EL kernel 4.18) refuse to execve()
# ELF binaries whose layout has been mutated by patchelf when patchelf had to
# grow DT_RPATH/DT_RUNPATH and therefore prepended a new PT_LOAD segment at a
# non-standard base address (e.g. 0x3ff000 with a read-write first PT_LOAD
# instead of the canonical 0x400000 read-only one). See
# https://github.com/ROCm/TheRock/issues/4271.
#
# To avoid this, every ELF TheRock produces has a "pad" entry appended to its
# INSTALL_RPATH at link time. The pad is a syntactically valid ORIGIN-relative
# path that is long enough to absorb any real RPATH value we later need to
# inject from the Python packaging layer. When py_packaging rewrites RPATHs
# with patchelf --set-rpath, the final string always fits inside the existing
# .dynstr allocation, so patchelf modifies the entry in place without touching
# program headers. The pad path itself points at a directory that never exists
# at runtime, so it has no effect on library resolution.
#
# The marker substring "__therock_patchelf_pad__" is intentional: CI greps for
# it in shipped wheels to detect any ELF where py_packaging failed to strip
# the pad, which would be a sign that the patchelf pass skipped that file.
set(THEROCK_INSTALL_RPATH_PAD_SIZE 1024 CACHE STRING
  "Byte length of the filler RPATH entry appended at link time so patchelf --set-rpath can overwrite in place (see issue #4271).")
set(THEROCK_INSTALL_RPATH_PAD_MARKER "__therock_patchelf_pad__" CACHE STRING
  "Literal substring embedded in the RPATH pad entry for CI detection of pad-stripping failures.")
mark_as_advanced(THEROCK_INSTALL_RPATH_PAD_SIZE THEROCK_INSTALL_RPATH_PAD_MARKER)

# Lazily build THEROCK_INSTALL_RPATH_PAD (CMake-list form, single entry) and
# THEROCK_INSTALL_RPATH_PAD_COLON (colon-joined form for consumers like
# LIBOMP_INSTALL_RPATH that expect a pre-joined POSIX rpath string). Both
# forms hold a single RPATH entry that is exactly
# THEROCK_INSTALL_RPATH_PAD_SIZE bytes long (minus trailing null).
function(_therock_compute_install_rpath_pad)
  if(DEFINED CACHE{THEROCK_INSTALL_RPATH_PAD})
    return()
  endif()
  set(_prefix "$ORIGIN/.${THEROCK_INSTALL_RPATH_PAD_MARKER}_")
  string(LENGTH "${_prefix}" _prefix_len)
  math(EXPR _fill_len "${THEROCK_INSTALL_RPATH_PAD_SIZE} - ${_prefix_len}")
  if(_fill_len LESS 1)
    message(FATAL_ERROR
      "THEROCK_INSTALL_RPATH_PAD_SIZE=${THEROCK_INSTALL_RPATH_PAD_SIZE} is too "
      "small to hold the marker prefix (${_prefix_len} bytes).")
  endif()
  string(REPEAT "X" "${_fill_len}" _filler)
  set(_pad "${_prefix}${_filler}")
  set(THEROCK_INSTALL_RPATH_PAD "${_pad}" CACHE INTERNAL
    "Single RPATH entry used to reserve .dynstr space for patchelf in-place overwrite (issue #4271).")
  set(THEROCK_INSTALL_RPATH_PAD_COLON "${_pad}" CACHE INTERNAL
    "Colon-joined form of THEROCK_INSTALL_RPATH_PAD for rpath variables that expect a POSIX rpath string.")
endfunction()
_therock_compute_install_rpath_pad()


# Sets a list of relative INSTALL_RPATH values on a target. This is a no-op
# outside of Posix systems.
# Args:
# TARGETS: Targets to modify INSTALL_RPATH of.
# PATHS: Origin-relative paths to set.
function(therock_set_install_rpath)
  cmake_parse_arguments(
    PARSE_ARGV 0 ARG
    ""
    ""
    "TARGETS;PATHS"
  )
  if(WIN32)
    return()
  endif()

  set(_rpath)
  foreach(path ${ARG_PATHS})
    if("${path}" STREQUAL ".")
      set(path_suffix "")
    else()
      set(path_suffix "/${path}")
    endif()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
      list(APPEND _rpath "@loader_path${path_suffix}")
    else()
      list(APPEND _rpath "$ORIGIN${path_suffix}")
    endif()
  endforeach()
  # Log the user-meaningful RPATH before appending the patchelf pad. The pad
  # is ~1KB of filler and dumping it into every "Set RPATH" status line would
  # drown out from-source build logs (issue #4271).
  set(_rpath_log_suffix "")
  if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin" AND THEROCK_INSTALL_RPATH_PAD)
    set(_rpath_log_suffix " (+patchelf pad)")
  endif()
  message(STATUS "Set RPATH ${_rpath}${_rpath_log_suffix} on ${ARG_TARGETS}")
  if(_rpath_log_suffix)
    list(APPEND _rpath "${THEROCK_INSTALL_RPATH_PAD}")
  endif()
  set_target_properties(${ARG_TARGETS} PROPERTIES INSTALL_RPATH "${_rpath}")
endfunction()


# Replaces a library linked with `-l` with a CMake target.
# Args:
# OLD_LIBRARY: Deprecated library linked with `-l`
# NEW_TARGET: New target library
function(therock_patch_linked_lib)
  cmake_parse_arguments(
    PARSE_ARGV 0 ARG
    ""
    ""
    "OLD_LIBRARY;NEW_TARGET"
  )
  therock_get_all_targets(all_targets "${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "Patching targets: ${all_targets}")
  foreach(target ${all_targets})
    get_target_property(link_libs "${target}" LINK_LIBRARIES)
    if("-l${ARG_OLD_LIBRARY}" IN_LIST link_libs)
      list(REMOVE_ITEM link_libs "-l${ARG_OLD_LIBRARY}")
      list(APPEND link_libs "${ARG_NEW_TARGET}")
      set_target_properties("${target}" PROPERTIES LINK_LIBRARIES "${link_libs}")
      message(WARNING "target ${target} depends on deprecated -l${ARG_OLD_LIBRARY}. Redirecting: ${link_libs}")
    endif()
  endforeach()
endfunction()
