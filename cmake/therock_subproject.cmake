# therock_subproject.cmake
# Facilities for defining build subprojects. This has some similarity to the
# built-in ExternalProject and FetchContent facilities, but it is intended to
# be performant and ergonomic for a super project of our scale where the sources
# of the subprojects are expected to be modified as part of the super-project
# development flow.

# Global properties.
# THEROCK_DEFAULT_CMAKE_VARS:
# List of CMake variables that will be injected by default into the
# project_init.cmake file of each subproject.
set_property(GLOBAL PROPERTY THEROCK_DEFAULT_CMAKE_VARS
  CMAKE_C_COMPILER
  CMAKE_CXX_COMPILER
  CMAKE_C_COMPILER_LAUNCHER
  CMAKE_CXX_COMPILER_LAUNCHER
  CMAKE_BUILD_TYPE
  CMAKE_INSTALL_LIBDIR
  CMAKE_PLATFORM_NO_VERSIONED_SONAME
  Python3_EXECUTABLE
  Python3_FIND_VIRTUALENV
  THEROCK_SOURCE_DIR
  ROCM_GIT_DIR
  ROCM_MAJOR_VERSION
  ROCM_MINOR_VERSION
  ROCM_PATCH_VERSION
  ROCM_VERSION
  AMDGPU_TARGETS
)

# therock_cmake_subproject_declare
# This declares a cmake based subproject by setting a number of key properties
# and setting up boiler-plate targets.
#
# Arguments:
# NAME: Globally unique subproject name. This will become the stem of various
# targets and therefore must be unique (even for nested projects) and a valid
# target identifier.
# ACTIVATE: Option to signify that this call should end by calling
# therock_cmake_subproject_activate. Do not specify this option if wishing to
# further configure the sub-project.
# SOURCE_DIR: Absolute path to the external source directory.
# DIR_PREFIX: By default, directories named "build", "install", "stamp" are
# created. But if there are multiple sub-projects in a parent dir, then they
# all must have a distinct prefix (not recommended).
# INSTALL_DESTINATION: Sub-directory within the install directory where this
# sub-project installs. Defaults to empty, meaning that it installs at the top
# of the namespace.
# CMAKE_ARGS: Additional CMake configure arguments.
# BUILD_DEPS: Projects which must build and provide their packages prior to this
# one.
# RUNTIME_DEPS: Projects which must build prior to this one and whose install
# files must be distributed with this project's artifacts in order to function.
function(therock_cmake_subproject_declare target_name)
  cmake_parse_arguments(
    PARSE_ARGV 1 ARG
    "ACTIVATE;EXCLUDE_FROM_ALL"
    "EXTERNAL_SOURCE_DIR;DIR_PREFIX;INSTALL_DESTINATION"
    "BUILD_DEPS;RUNTIME_DEPS;CMAKE_ARGS"
  )
  if(TARGET "${target_name}")
    message(FATAL_ERROR "Cannot declare subproject '${target_name}': a target with that name already exists")
  endif()

  message(STATUS "Including subproject ${target_name} (from ${ARG_EXTERNAL_SOURCE_DIR})")
  add_custom_target("${target_name}" COMMENT "Top level target to build the ${target_name} sub-project")

  # Build directory.
  set(_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/${ARG_DIR_PREFIX}build")
  make_directory("${_binary_dir}")

  # Install directory.
  set(_install_dir "${CMAKE_CURRENT_BINARY_DIR}/${ARG_DIR_PREFIX}install")
  make_directory("${_install_dir}")

  # Dist directory.
  set(_dist_dir "${CMAKE_CURRENT_BINARY_DIR}/${ARG_DIR_PREFIX}dist")
  make_directory("${_dist_dir}")

  # Stamp directory.
  set(_stamp_dir "${CMAKE_CURRENT_BINARY_DIR}/${ARG_DIR_PREFIX}stamp")
  make_directory("${_stamp_dir}")

  set_target_properties("${target_name}" PROPERTIES
    THEROCK_SUBPROJECT cmake
    THEROCK_EXCLUDE_FROM_ALL "${ARG_EXCLUDE_FROM_ALL}"
    THEROCK_EXTERNAL_SOURCE_DIR "${ARG_EXTERNAL_SOURCE_DIR}"
    THEROCK_BINARY_DIR "${_binary_dir}"
    THEROCK_DIST_DIR "${_dist_dir}"
    THEROCK_INSTALL_DIR "${_install_dir}"
    THEROCK_INSTALL_DESTINATION "${ARG_INSTALL_DESTINATION}"
    THEROCK_STAMP_DIR "${_stamp_dir}"
    THEROCK_CMAKE_SOURCE_DIR "${ARG_EXTERNAL_SOURCE_DIR}"
    THEROCK_CMAKE_PROJECT_INIT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${ARG_BUILD_DIR}_init.cmake"
    THEROCK_CMAKE_ARGS "${ARG_CMAKE_ARGS}"
    THEROCK_BUILD_DEPS "${ARG_BUILD_DEPS}"
    THEROCK_RUNTIME_DEPS "${ARG_RUNTIME_DEPS}"
  )

  if(ARG_ACTIVATE)
    therock_cmake_subproject_activate("${target_name}")
  endif()
endfunction()

# therock_cmake_subproject_provide_package
# Declares that a subproject provides a given package which should be findable
# with `find_package(package_name)` at the given path relative to its install
# directory.
function(therock_cmake_subproject_provide_package target_name package_name relative_path)
  string(APPEND CMAKE_MESSAGE_INDENT "  ")
  get_target_property(_existing_packages "${target_name}" THEROCK_PROVIDE_PACKAGES)
  if(${package_name} IN_LIST _existing_packages)
    message(FATAL_ERROR "Package defined multiple times on sub-project ${target_name}: ${package_name}")
  endif()
  set_property(TARGET "${target_name}" APPEND PROPERTY THEROCK_PROVIDE_PACKAGES "${package_name}")
  set(_relpath_name THEROCK_PACKAGE_RELPATH_${package_name})
  set_property(TARGET "${target_name}" PROPERTY ${_relpath_name} "${relative_path}")
  message(STATUS "PROVIDE ${package_name} = ${relative_path} (from ${target_name})")
endfunction()

# therock_cmake_subproject_activate
# If using multi-step setup (i.e. without 'ACTIVATE' on the declare), then this
# must be called once all configuration is complete.
function(therock_cmake_subproject_activate target_name)
  # Make sure it is a sub-project.
  get_target_property(_is_subproject "${target_name}" THEROCK_SUBPROJECT)
  if(NOT _is_subproject STREQUAL "cmake")
    message(FATAL_ERROR "Target ${target_name} is not a sub-project")
  endif()

  # Get properties.
  get_target_property(_binary_dir "${target_name}" THEROCK_BINARY_DIR)
  get_target_property(_build_deps "${target_name}" THEROCK_BUILD_DEPS)
  get_target_property(_dist_dir "${target_name}" THEROCK_DIST_DIR)
  get_target_property(_runtime_deps "${target_name}" THEROCK_RUNTIME_DEPS)
  get_target_property(_cmake_args "${target_name}" THEROCK_CMAKE_ARGS)
  get_target_property(_cmake_project_init_file "${target_name}" THEROCK_CMAKE_PROJECT_INIT_FILE)
  get_target_property(_cmake_source_dir "${target_name}" THEROCK_CMAKE_SOURCE_DIR)
  get_target_property(_exclude_from_all "${target_name}" THEROCK_EXCLUDE_FROM_ALL)
  get_target_property(_external_source_dir "${target_name}" THEROCK_EXTERNAL_SOURCE_DIR)
  get_target_property(_install_destination "${target_name}" THEROCK_INSTALL_DESTINATION)
  get_target_property(_install_dir "${target_name}" THEROCK_INSTALL_DIR)
  get_target_property(_sources "${target_name}" SOURCES)
  get_target_property(_stamp_dir "${target_name}" THEROCK_STAMP_DIR)

  # Handle optional properties.
  if(NOT _sources)
    set(_sources)
  endif()

  # Generate the project_init.cmake
  set(_dep_provider_file)
  if(_build_deps OR _runtime_deps)
    set(_dep_provider_file "${THEROCK_SOURCE_DIR}/cmake/therock_subproject_dep_provider.cmake")
  endif()
  set(_injected_file "${THEROCK_SOURCE_DIR}/cmake/therock_external_project_include.cmake")
  get_property(_mirror_cmake_vars GLOBAL PROPERTY THEROCK_DEFAULT_CMAKE_VARS)
  set(_init_contents)
  foreach(_var_name ${_mirror_cmake_vars})
    string(APPEND _init_contents "set(${_var_name} \"@${_var_name}@\" CACHE STRING \"\" FORCE)\n")
  endforeach()
  _therock_cmake_subproject_setup_deps(_deps_contents ${_build_deps} ${_runtime_deps})
  string(APPEND _init_contents "${_deps_contents}")
  if(_dep_provider_file)
    string(APPEND _init_contents "include(${_dep_provider_file})\n")
  endif()
  string(APPEND _init_contents "include(${_injected_file})\n")
  file(CONFIGURE OUTPUT "${_cmake_project_init_file}" CONTENT "${_init_contents}" @ONLY ESCAPE_QUOTES)

  # Transform build and run deps from target form (i.e. 'ROCR-Runtime' to a dependency
  # on the stage.stamp file). These are a dependency for configure.
  _therock_cmake_subproject_deps_to_stamp(_configure_dep_stamps stage.stamp ${_build_deps} ${runtime_deps})

  # Target flags.
  set(_all_option)
  if(NOT _exclude_from_all)
    set(_all_option "ALL")
  endif()

  # configure target
  set(_configure_stamp_file "${_stamp_dir}/configure.stamp")
  set(_terminal_option)
  if(THEROCK_INTERACTIVE)
    set(_terminal_option "USES_TERMINAL")
  endif()
  set(_install_destination_dir "${_install_dir}")
  if(_install_destination)
    cmake_path(APPEND _install_destination_dir "${_install_destination}")
  endif()
  add_custom_command(
    OUTPUT "${_configure_stamp_file}"
    COMMAND "${CMAKE_COMMAND}"
      "-G${CMAKE_GENERATOR}"
      "-B${_binary_dir}"
      "-S${_cmake_source_dir}"
      "-DCPACK_PACKAGING_INSTALL_PREFIX=${STAGING_INSTALL_DIR}"
      "-DCMAKE_INSTALL_PREFIX=${_install_destination_dir}"
      "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${_cmake_project_init_file}"
      ${_cmake_args}
    COMMAND "${CMAKE_COMMAND}" -E touch "${_configure_stamp_file}"
    WORKING_DIRECTORY "${_binary_dir}"
    COMMENT "Configure sub-project ${target_name}"
    BYPRODUCTS
      "${_binary_dir}/CMakeCache.txt"
      "${_binary_dir}/cmake_install.cmake"
    DEPENDS
      "${_cmake_source_dir}/CMakeLists.txt"
      "${_cmake_project_init_file}"
      "${_injected_file}"
      ${_dep_provider_file}
      ${_configure_dep_stamps}

      # TODO: Have a mechanism for adding more depends for better rebuild ergonomics
    ${_terminal_option}
  )
  add_custom_target(
    "${target_name}+configure"
    ${_all_option}
    DEPENDS "${_configure_stamp_file}"
  )
  add_dependencies("${target_name}" "${target_name}+configure")

  # build target.
  set(_build_stamp_file "${_stamp_dir}/build.stamp")
  add_custom_command(
    OUTPUT "${_build_stamp_file}"
    COMMAND "${CMAKE_COMMAND}" "--build" "${_binary_dir}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_build_stamp_file}"
    WORKING_DIRECTORY "${_binary_dir}"
    COMMENT "Building sub-project ${target_name}"
    DEPENDS
      "${_configure_stamp_file}"
      ${_sources}
    USES_TERMINAL  # Always use the terminal for the build as we want to serialize
  )
  add_custom_target(
    "${target_name}+build"
    ${_all_option}
    DEPENDS
      "${_build_stamp_file}"
  )
  add_dependencies("${target_name}" "${target_name}+build")

  # stage install target.
  set(_stage_stamp_file "${_stamp_dir}/stage.stamp")
  add_custom_command(
    OUTPUT "${_stage_stamp_file}"
    COMMAND "${CMAKE_COMMAND}" --install "${_binary_dir}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${_stage_stamp_file}"
    WORKING_DIRECTORY "${_binary_dir}"
    COMMENT "Stage installing sub-project ${target_name}"
    DEPENDS
      "${_build_stamp_file}"
    ${_terminal_option}
  )
  add_custom_target(
    "${target_name}+stage"
    ${_all_option}
    DEPENDS
      "${_stage_stamp_file}"
  )
  add_dependencies("${target_name}" "${target_name}+stage")

  # dist install target.
  set(_dist_stamp_file "${_stamp_dir}/dist.stamp")
  set(_link_dist_script "${THEROCK_SOURCE_DIR}/build_tools/link_dist_dir.cmake")
  _therock_cmake_subproject_get_install_dirs(
    _dist_source_dirs "${target_name}" ${_runtime_deps})
  add_custom_command(
    OUTPUT "${_dist_stamp_file}"
    COMMAND "${CMAKE_COMMAND}" -P "${_link_dist_script}" "${_dist_dir}" ${_dist_source_dirs}
    COMMAND "${CMAKE_COMMAND}" -E touch "${_dist_stamp_file}"
    COMMENT "Linking sub-project dist directory for ${target_name}"
    DEPENDS
      "${_stage_stamp_file}"
      "${_link_dist_script}"
    ${_terminal_option}
  )
  add_custom_target(
    "${target_name}+dist"
    ${_all_option}
    DEPENDS
      "${_dist_stamp_file}"
  )
  add_dependencies("${target_name}" "${target_name}+dist")

  # expunge target
  add_custom_target(
    "${target_name}+expunge"
    COMMAND
      ${CMAKE_COMMAND} -E rm -rf "${_binary_dir}" "${_install_dir}" "${_stamp_dir}" "${_dist_dir}"
  )
endfunction()

# therock_cmake_subproject_glob_c_sources
# Adds C/C++ sources from given project subdirectories to the list of sources for
# a sub-project. This allows the super-project build system to know when to
# re-trigger the build step of the sub-project. There are many issues with globs
# in CMake, but as an ergonomic win, this is deemed an acceptable compromise
# to a large degree of explicitness.
function(therock_cmake_subproject_glob_c_sources target_name)
  cmake_parse_arguments(
    PARSE_ARGV 1 ARG
    ""
    ""
    "SUBDIRS"
  )
  get_target_property(_project_source_dir "${target_name}" THEROCK_EXTERNAL_SOURCE_DIR)
  set(_globs)
  foreach(_subdir ${ARG_SUBDIRS})
    set(_s "${_project_source_dir}/${_subdir}")
    list(APPEND _globs
      "${_s}/*.h"
      "${_s}/*.hpp"
      "${_s}/*.inc"
      "${_s}/*.cc"
      "${_s}/*.cpp"
      "${_s}/*.c"
    )
  endforeach()
  file(GLOB_RECURSE _files LIST_DIRECTORIES FALSE
    CONFIGURE_DEPENDS
    ${_globs}
  )
  target_sources("${target_name}" PRIVATE ${_files})
endfunction()

# Builds a CMake language fragment to set up a dependency provider such that
# it handles super-project provided dependencies locally.
function(_therock_cmake_subproject_setup_deps out_contents)
  string(APPEND CMAKE_MESSAGE_INDENT "  ")
  set(_contents "set(THEROCK_PROVIDED_PACKAGES)\n")
  foreach(dep_target ${ARGN})
    get_target_property(_is_subproject "${dep_target}" THEROCK_SUBPROJECT)
    if(NOT _is_subproject STREQUAL "cmake")
      message(FATAL_ERROR "Target ${target_name} is not a sub-project but was referenced as a dependency")
    endif()

    get_target_property(_provides "${dep_target}" THEROCK_PROVIDE_PACKAGES)
    if(_provides)
      foreach(_package_name ${_provides})
        get_target_property(_install_dir "${dep_target}" THEROCK_INSTALL_DIR)
        set(_relpath_name THEROCK_PACKAGE_RELPATH_${_package_name})
        get_target_property(_relpath "${dep_target}" ${_relpath_name})
        if(NOT _install_dir OR NOT _relpath)
          message(FATAL_ERROR "Missing package info props for ${_package_name} on ${dep_target}: '${_install_dir}' ${_relpath_name}='${_relpath}'")
        endif()
        cmake_path(APPEND _install_dir "${_relpath}" OUTPUT_VARIABLE _find_package_path)
        message(STATUS "INJECT ${_package_name} = ${_find_package_path} (from ${dep_target})")
        string(APPEND _contents "set(THEROCK_PACKAGE_DIR_${_package_name} \"${_find_package_path}\")\n")
        string(APPEND _contents "list(APPEND THEROCK_PROVIDED_PACKAGES ${_package_name})\n")
      endforeach()
    endif()
  endforeach()
  set("${out_contents}" "${_contents}" PARENT_SCOPE)
endfunction()

# Gets the staging install directories for a list of subproject deps.
function(_therock_cmake_subproject_get_install_dirs out_dirs)
  set(_dirs)
  foreach(target_name ${ARGN})
    get_target_property(_install_dir "${target_name}" THEROCK_INSTALL_DIR)
    if(NOT _install_dir)
      message(FATAL_ERROR "Sub-project target ${target_name} does not have an install dir")
    endif()
    list(APPEND _dirs "${_install_dir}")
  endforeach()
  set(${out_dirs} "${_dirs}" PARENT_SCOPE)
endfunction()

# Transforms a list of sub-project targets to corresponding stamp files of
# `stamp_name`. These are the actual build system deps that are encoded in the
# commands (whereas the target names are just for humans).
function(_therock_cmake_subproject_deps_to_stamp out_stamp_files stamp_name)
  set(_stamp_files)
  foreach(target_name ${ARGN})
    # Make sure it is a sub-project.
    get_target_property(_is_subproject "${target_name}" THEROCK_SUBPROJECT)
    if(NOT _is_subproject STREQUAL "cmake")
      message(FATAL_ERROR "Target ${target_name} is not a sub-project")
    endif()
    get_target_property(_stamp_dir "${target_name}" THEROCK_STAMP_DIR)
    if(NOT _stamp_dir)
      message(FATAL_ERROR "Sub-project is missing a stamp dir: ${target_name}")
    endif()

    list(APPEND _stamp_files "${_stamp_dir}/${stamp_name}")
  endforeach()
  set(${out_stamp_files} "${_stamp_files}" PARENT_SCOPE)
endfunction()
