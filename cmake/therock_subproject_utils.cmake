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


# Extracts a dependency prefix path from a path list (typically CMAKE_PREFIX_PATH)
# by filtering for a specific pattern. This is useful for autotools-based builds
# that require explicit --with-* configure flags, as autotools doesn't use
# CMAKE_PREFIX_PATH directly.
# Args:
#   input_path_list: Name of the variable containing the list of paths (e.g., "CMAKE_PREFIX_PATH")
#   pattern: Regex pattern to filter the paths (e.g., "/gmp/build/stage")
#   output_var: Name of the variable to store the result in
# Example:
#   fetch_library_prefix(CMAKE_PREFIX_PATH "/gmp/build/stage" GMP_PREFIX)
#   # Sets GMP_PREFIX to the first path in CMAKE_PREFIX_PATH matching the pattern
function(therock_fetch_library_prefix input_path_list pattern output_var)
  # Access the list contents using the input variable name
  set(_input_list "${${input_path_list}}")
  # Filter the list to include only items matching the pattern
  list(FILTER _input_list INCLUDE REGEX "${pattern}")
  # Check if a match was found
  if(_input_list)
    # Get the first match and return it
    list(GET _input_list 0 _result_path)
    set(${output_var} "${_result_path}" PARENT_SCOPE)
    message(STATUS "Extracted path for pattern '${pattern}': ${_result_path}")
  else()
    # Error if no match found
    message(SEND_ERROR "Did not find a path containing '${pattern}' in ${input_path_list}.")
  endif()
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
