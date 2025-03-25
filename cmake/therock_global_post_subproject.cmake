# This file is called as a hook at the end of every sub-project's configure.
# It performs additional super-project level fixups and invokes the user's
# post hook if applicable (see THEROCK_USER_POST_HOOK).

# Make project-wide utilities available to the post hook.
list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_subproject_utils)

# Do post processing on all defined targets.
set(THEROCK_ALL_TARGETS)
therock_get_all_targets(THEROCK_ALL_TARGETS "${CMAKE_CURRENT_SOURCE_DIR}")

# Delegate to user post hook.
if(THEROCK_USER_POST_HOOK)
  include("${THEROCK_USER_POST_HOOK}")
endif()

# Performs post-processing on a dynamically linked target (executables and
# shared libraries) for rpath settings.
# Unless if disabled by the global NO_INSTALL_RPATH on the project or locally
# via THEROCK_NO_INSTALL_RPATH target property, performs default installation
# RPATH assignment.
function(_therock_post_process_rpath_target target target_type)
  get_target_property(_no_install_rpath "${target}" THEROCK_NO_INSTALL_RPATH)
  if(THEROCK_NO_INSTALL_RPATH OR _no_install_rpath)
    return()
  endif()

  # Determine the target's origin, which is what RPATHs must be relative to.
  get_target_property(_origin ${target} THEROCK_INSTALL_RPATH_ORIGIN)
  if(NOT _origin)
    if(target_type STREQUAL "EXECUTABLE")
      set(_origin "${THEROCK_INSTALL_RPATH_EXECUTABLE_DIR}")
    elseif(target_type STREQUAL "SHARED_LIBRARY")
      set(_origin "${THEROCK_INSTALL_RPATH_LIBRARY_DIR}")
    else()
      message(FATAL_ERROR "Unhandled target type ${target_type}")
    endif()
  endif()

  # Now need to create a path relative to each origin.
  set(_install_rpath)
  foreach(rpath_dir ${THEROCK_PRIVATE_INSTALL_RPATH_DIRS})
    cmake_path(RELATIVE_PATH rpath_dir BASE_DIRECTORY ${_origin})
    list(APPEND _install_rpath "${rpath_dir}")
  endforeach()

  therock_set_install_rpath(TARGETS "${target}" PATHS ${_install_rpath})
endfunction()

# Iterate over all targets and set default RPATH (unless if globally disabled
# for the subproject).
block()
  if(NOT THEROCK_NO_INSTALL_RPATH)
    foreach(target ${THEROCK_ALL_TARGETS})
      get_target_property(target_type "${target}" TYPE)
      get_target_property(target_alias "${target}" ALIASED_TARGET)
      if(ALIASED_TARGET OR
        (NOT target_type STREQUAL "SHARED_LIBRARY" AND NOT target_type STREQUAL "EXECUTABLE"))
        continue()
      endif()

      _therock_post_process_rpath_target(${target} ${target_type})
    endforeach()
  endif()
endblock()
