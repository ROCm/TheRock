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
  message(STATUS "Set RPATH ${_rpath} on ${ARG_TARGETS}")
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


# RCCL force-sets Release flags to -O3 after the toolchain file, which drops
# host -Oz. That overwrite happens before add_subdirectory, so put the flag
# back at the end of this directory. Compile rules read the cache at generate
# time. Do not re-add -DNDEBUG: RCCL drops it on purpose.
function(therock_asan_compact_restore_host_oz)
  if(NOT THEROCK_ASAN_COMPACT)
    return()
  endif()
  foreach(_var IN ITEMS CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE)
    if("${${_var}}" MATCHES "(^|[ \t])-Xarch_host[ \t]+-Oz([ \t]|$)")
      continue()
    endif()
    set("${_var}" "${${_var}} -Xarch_host -Oz" CACHE STRING "Release flags" FORCE)
  endforeach()
endfunction()


# Re-apply host -Oz after this subproject's CMakeLists overwrites Release flags.
function(therock_asan_compact_preserve_host_oz)
  if(NOT THEROCK_ASAN_COMPACT)
    return()
  endif()
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL therock_asan_compact_restore_host_oz)
endfunction()


# --offload-compress for HIP code compiled as C++ via hip::device.
# CMAKE_HIP_FLAGS never applies to those targets, and a global C++ flag warns
# on ordinary C++ files.
function(therock_asan_compact_add_offload_compress dir)
  if(NOT THEROCK_ASAN_COMPACT)
    return()
  endif()
  therock_get_all_targets(_targets "${dir}")
  foreach(_tgt IN LISTS _targets)
    if(NOT TARGET "${_tgt}")
      continue()
    endif()
    get_target_property(_type "${_tgt}" TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY" OR _type STREQUAL "UTILITY")
      continue()
    endif()
    get_target_property(_libs "${_tgt}" LINK_LIBRARIES)
    get_target_property(_iface "${_tgt}" INTERFACE_LINK_LIBRARIES)
    set(_joined "${_libs};${_iface}")
    if(NOT _joined MATCHES "(^|;)hip::device(;|$)")
      continue()
    endif()
    # Target option, not a global flag string, so it is not copied into
    # custom commands that cannot evaluate generator expressions.
    target_compile_options("${_tgt}" PRIVATE "$<$<COMPILE_LANGUAGE:CXX,HIP>:--offload-compress>")
    message(STATUS "ASAN compact: --offload-compress on ${_tgt} (links hip::device)")
  endforeach()
endfunction()
