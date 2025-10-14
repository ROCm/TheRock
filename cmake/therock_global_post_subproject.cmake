# This file is called as a hook at the end of every sub-project's configure.
# It performs additional super-project level fixups and invokes the user's
# post hook if applicable (see THEROCK_USER_POST_HOOK).

# Make project-wide utilities available to the post hook.
list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_subproject_utils)

# Do post processing on all defined targets.
set(THEROCK_ALL_TARGETS)
therock_get_all_targets(THEROCK_ALL_TARGETS "${CMAKE_CURRENT_SOURCE_DIR}")

# Separate linked targets into categories.
set(THEROCK_EXECUTABLE_TARGETS)
set(THEROCK_SHARED_LIBRARY_TARGETS)
set(THEROCK_MODULE_TARGETS)
block(SCOPE_FOR VARIABLES)
  foreach(target ${THEROCK_ALL_TARGETS})
    get_target_property(target_type "${target}" TYPE)
    get_target_property(target_alias "${target}" ALIASED_TARGET)
    if(target_alias)
      continue()
    endif()
    if("${target_type}" STREQUAL "SHARED_LIBRARY")
      list(APPEND THEROCK_SHARED_LIBRARY_TARGETS "${target}")
    elseif("${target_type}" STREQUAL "EXECUTABLE")
      list(APPEND THEROCK_EXECUTABLE_TARGETS "${target}")
    elseif("${target_type}" STREQUAL "MODULE_LIBRARY")
      list(APPEND THEROCK_MODULE_TARGETS "${target}")
    endif()
  endforeach()

  set(THEROCK_SHARED_LIBRARY_TARGETS "${THEROCK_SHARED_LIBRARY_TARGETS}" PARENT_SCOPE)
  set(THEROCK_EXECUTABLE_TARGETS "${THEROCK_EXECUTABLE_TARGETS}" PARENT_SCOPE)
  set(THEROCK_MODULE_TARGETS "${THEROCK_MODULE_TARGETS}" PARENT_SCOPE)
endblock()

# Delegate to user post hook.
if(THEROCK_USER_POST_HOOK)
  include("${THEROCK_USER_POST_HOOK}")
endif()

# Performs post-processing on a dynamically linked target (executables and
# shared libraries) for rpath settings.
# Unless if disabled by the global NO_INSTALL_RPATH on the project or locally
# via THEROCK_NO_INSTALL_RPATH target property, performs default installation
# RPATH assignment.
function(_therock_post_process_rpath_target target)
  if(THEROCK_PRIVATE_BUILD_RPATH_DIRS)
    message(STATUS "Add build RPATH ${THEROCK_PRIVATE_BUILD_RPATH_DIRS} on ${target}")
    set_property(TARGET "${target}" APPEND PROPERTY BUILD_RPATH "${THEROCK_PRIVATE_BUILD_RPATH_DIRS}")
  endif()
  get_target_property(_no_install_rpath "${target}" THEROCK_NO_INSTALL_RPATH)
  if(THEROCK_NO_INSTALL_RPATH OR _no_install_rpath)
    return()
  endif()
  get_target_property(target_type "${target}" TYPE)

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


# Iterate over all shared library and executable targets and set default RPATH
# (unless if globally disabled for the subproject).
block(SCOPE_FOR VARIABLES)
  if(NOT THEROCK_NO_INSTALL_RPATH)
    foreach(target ${THEROCK_EXECUTABLE_TARGETS} ${THEROCK_SHARED_LIBRARY_TARGETS})
      _therock_post_process_rpath_target(${target})
    endforeach()
  endif()
endblock()

# Process all shared library and executable targets and emit install time code
# to process their build id and split debug files out.
if(THEROCK_SPLIT_DEBUG_INFO AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(CMakeFindBinUtils)
  block(SCOPE_FOR POLICIES VARIABLES)
    install(
      CODE "set(THEROCK_DEBUG_BUILD_ID_PATHS)"
      CODE "set(THEROCK_OBJCOPY \"${CMAKE_OBJCOPY}\")"
      CODE "set(THEROCK_READELF \"${CMAKE_READELF}\")"
      CODE "set(THEROCK_STAGE_INSTALL_ROOT \"${THEROCK_STAGE_INSTALL_ROOT}\")"
      COMPONENT THEROCK_DEBUG_BUILD_ID
    )
    foreach(target
            ${THEROCK_EXECUTABLE_TARGETS}
            ${THEROCK_SHARED_LIBRARY_TARGETS}
            ${THEROCK_MODULE_TARGETS})
      message(STATUS "Splitting debug info for ${target}")
      set(_target_path "$<TARGET_FILE:${target}>")
      install(
        CODE "list(APPEND THEROCK_DEBUG_BUILD_ID_PATHS \"${_target_path}\")"
        COMPONENT THEROCK_DEBUG_BUILD_ID
      )
      # Must be built with build-id enabled in order to do debug symbol sep.
      target_link_options("${target}" PRIVATE "-Wl,--build-id")
    endforeach()
    install(
        SCRIPT "${THEROCK_SOURCE_DIR}/cmake/therock_install_linux_build_id_files.cmake"
        COMPONENT THEROCK_DEBUG_BUILD_ID
    )
  endblock()
endif()

# Add convenience targets within the sub-project that interact with the super-project.
# This allows developers to work entirely in the sub-project build directory and
# perform most operations.

# Removes the sub-project stage.stamp file. This will cause any subsequent action
# on the super-project to see the sub-project as out of date with respect to
# being stage installed and populated in distributions. We invoke it on ALL
# so that most local build commands (i.e. `ninja` without args) triggers
# invalidation. During normal super-project build, once the build phase is
# done, the build.stamp will be touched, which will cause the stage.stamp to
# show as out of date. So either way, the ordering is preserved.
add_custom_target(therock-touch ALL
  COMMAND
    "${CMAKE_COMMAND}" -E rm -f "${THEROCK_STAGE_STAMP_FILE}"
  VERBATIM
)

# Removes the sub-project build.stamp file, indicating that the project must be
# rebuilt, then invokes the parent {target}+dist, causing all stage installation
# and distributions to be populated upon successful build.
add_custom_target(therock-dist
  COMMAND
    "${CMAKE_COMMAND}" -E rm -f "${THEROCK_BUILD_STAMP_FILE}"
  COMMAND
    "${CMAKE_COMMAND}" --build "${THEROCK_BINARY_DIR}" --target "${THEROCK_SUBPROJECT_TARGET}+dist"
  COMMENT "Trigger super-project dist"
  VERBATIM
  USES_TERMINAL
)
