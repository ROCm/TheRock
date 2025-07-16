# therock_subproject.cmake
# Facilities for defining build subprojects. This has some similarity to the
# built-in ExternalProject and FetchContent facilities, but it is intended to
# be performant and ergonomic for a super project of our scale where the sources
# of the subprojects are expected to be modified as part of the super-project
# development flow.

include(ExternalProject)

# Global properties.
# THEROCK_DEFAULT_CMAKE_VARS:
# List of CMake variables that will be injected by default into the
# project_init.cmake file of each subproject.
set_property(GLOBAL PROPERTY THEROCK_DEFAULT_CMAKE_VARS
  CMAKE_PROGRAM_PATH
  CMAKE_PLATFORM_NO_VERSIONED_SONAME
  Python3_EXECUTABLE
  Python3_FIND_VIRTUALENV
  THEROCK_SOURCE_DIR
  THEROCK_BINARY_DIR
  THEROCK_BUILD_TESTING
  THEROCK_USE_SAFE_DEPENDENCY_PROVIDER
  ROCM_SYMLINK_LIBS

  # RPATH handling.
  THEROCK_NO_INSTALL_RPATH
  THEROCK_PRIVATE_INSTALL_RPATH_DIRS
  THEROCK_INSTALL_RPATH_EXECUTABLE_DIR
  THEROCK_INSTALL_RPATH_LIBRARY_DIR

  # Debug info handling.
  THEROCK_SPLIT_DEBUG_INFO
)

# Whenever a new package is advertised by the super-project, it is added here.
# This is used by the sub-project dependency resolver to error if a package
# dep is added but not declared (i.e. if it would fall back to the system
# resolver).
set_property(GLOBAL PROPERTY THEROCK_ALL_PROVIDED_PACKAGES)

# Some sub-projects do not react well to not having any GPU targets to build.
# In this case, we build them with a default target. This should only happen
# with target filtering for non-production, single target builds, and we will
# warn about it.
set(THEROCK_SUBPROJECTS_REQUIRING_DEFAULT_GPU_TARGETS hipBLASLt)
set(THEROCK_DEFAULT_GPU_TARGETS "gfx1100")

set_property(GLOBAL PROPERTY THEROCK_SUBPROJECT_COMPILE_COMMANDS_FILES)

if(CMAKE_C_VISIBILITY_PRESET)
  list(APPEND THEROCK_DEFAULT_CMAKE_VARS ${CMAKE_C_VISIBILITY_PRESET})
endif()
if(CMAKE_CXX_VISIBILITY_PRESET)
  list(APPEND THEROCK_DEFAULT_CMAKE_VARS ${CMAKE_CXX_VISIBILITY_PRESET})
endif()

# CXX flags that we hard-code in the toolchain file when building projects
# that use amd-llvm. In these cases, because it is our built toolchain, we
# don't need to probe warning flag availability and just have a hard-coded
# list. We only squelch warnings here that do not signal code correctness
# issues.
# TODO: Clean up warning flags (https://github.com/ROCm/TheRock/issues/47)
set(THEROCK_AMD_LLVM_DEFAULT_CXX_FLAGS
  -Wno-documentation-unknown-command
  -Wno-documentation-pedantic
  -Wno-unused-command-line-argument

  # There are real issues here but rocBLAS and rocSPARSE generate 300-400MB of
  # warning logs with this enabled, practically breaking build tooling.
  -Wno-explicit-specialization-storage-class
)

if(WIN32)
  # TODO(#36): Could we fix these warnings as part of enabling shared library builds?
  # These are frequently set in subproject toolchain-windows.cmake files.
  # Warning example:
  #     __declspec attribute 'dllexport' is not supported
  list(APPEND THEROCK_AMD_LLVM_DEFAULT_CXX_FLAGS -Wno-ignored-attributes)
  # Warning example:
  #     unknown attribute '__dllimport__' ignored
  list(APPEND THEROCK_AMD_LLVM_DEFAULT_CXX_FLAGS -Wno-unknown-attributes)

  # Warning example:
  #     duplicate 'static' declaration specifier
  list(APPEND THEROCK_AMD_LLVM_DEFAULT_CXX_FLAGS -Wno-duplicate-decl-specifier)
endif()

# Generates a command prefix that can be prepended to any custom command line
# to perform log/console redirection and pretty printing.
# LOG_FILE: If given, command output will also be sent to this log file. If
#   not absolute, it will be made absolute relative to logs/ in the project
#   binary directory.
# OUTPUT_ON_FAILURE: Boolean value to indicate whether output should go to the
#   console only on failure.
# LABEL: Label to prefix console output with.
#
# This uses the build_tools/teatime.py script for output management. See that
# script for further details. One thing to note: if TEATIME_LABEL_GH_GROUP=1
# in the environment, console output will be formatted with GitHub action
# begin/end group markers vs log line prefixes. Generally, you want to set
# this in CI jobs.
function(therock_subproject_log_command out_var)
  cmake_parse_arguments(
    PARSE_ARGV 1 ARG
    ""
    "LOG_FILE;LABEL;OUTPUT_ON_FAILURE"
    ""
  )

  set(command
    "${Python3_EXECUTABLE}"
    "${THEROCK_SOURCE_DIR}/build_tools/teatime.py"
    "--log-timestamps"
  )
  if(ARG_LABEL)
    list(APPEND command "--label" "${ARG_LABEL}")
  endif()
  if(ARG_OUTPUT_ON_FAILURE)
    list(APPEND command "--no-interactive")
  else()
    list(APPEND command "--interactive")
  endif()
  if(ARG_LOG_FILE)
    cmake_path(ABSOLUTE_PATH ARG_LOG_FILE BASE_DIRECTORY "${THEROCK_BINARY_DIR}/logs")
    list(APPEND command "${ARG_LOG_FILE}")
  endif()
  list(APPEND command "--")

  set("${out_var}" "${command}" PARENT_SCOPE)
endfunction()

# therock_subproject_fetch
# Fetches arbitrary content. This mostly defers to ExternalProject_Add to get
# content but it performs no actual building.
# All unrecognized options are passed to ExternalProject_Add.
# This can interoperate with therock_cmake_subproject_declare by adding the
# CMAKE_PROJECT option, which makes the CMakeLists.txt in the archive visible
# to CMake (which the subproject depends on). Additional touch byproducts
# can be generated with TOUCH.
function(therock_subproject_fetch target_name)
  cmake_parse_arguments(
    PARSE_ARGV 1 ARG
    "CMAKE_PROJECT"
    "SOURCE_DIR;PREFIX;EXCLUDE_FROM_ALL"
    "TOUCH"
  )

  if(NOT DEFINED ARG_EXCLUDE_FROM_ALL)
    set(ARG_EXCLUDE_FROM_ALL TRUE)
  endif()
  if(NOT DEFINED ARG_PREFIX)
    set(ARG_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_fetch")
  endif()
  if(NOT DEFINED ARG_SOURCE_DIR)
    set(ARG_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source")
  endif()

  set(_extra)
  # In order to interop with therock_cmake_subproject_declare, the CMakeLists.txt
  # file must exist so we mark this as a by-product. This serves as the dependency
  # anchor and causes proper ordering of fetch->configure.
  if(ARG_CMAKE_PROJECT)
    list(APPEND ARG_TOUCH "${ARG_SOURCE_DIR}/CMakeLists.txt")
  endif()
  if(ARG_TOUCH)
    list(APPEND _extra
      BUILD_COMMAND "${CMAKE_COMMAND}" -E touch ${ARG_TOUCH}
      BUILD_BYPRODUCTS ${ARG_TOUCH}
    )
  else()
    list(APPEND _extra "BUILD_COMMAND" "")
  endif()

  ExternalProject_Add(
    "${target_name}"
    EXCLUDE_FROM_ALL "${ARG_EXCLUDE_FROM_ALL}"
    PREFIX "${ARG_PREFIX}"
    DOWNLOAD_NO_PROGRESS ON
    LOG_DOWNLOAD ON
    LOG_MERGED_STDOUTERR ON
    LOG_OUTPUT_ON_FAILURE ON
    SOURCE_DIR "${ARG_SOURCE_DIR}"
    CONFIGURE_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
    ${_extra}
    ${ARG_UNPARSED_ARGUMENTS}
  )
endfunction()

# therock_cmake_subproject_declare
# This declares a cmake based subproject by setting a number of key properties
# and setting up boiler-plate targets.
#
# Arguments:
# NAME: Globally unique subproject name. This will become the stem of various
#   targets and therefore must be unique (even for nested projects) and a valid
#   target identifier.
# ACTIVATE: Option to signify that this call should end by calling
#   therock_cmake_subproject_activate. Do not specify this option if wishing to
#   further configure the sub-project.
# USE_DIST_AMDGPU_TARGETS: Use the dist GPU targets vs the shard specific GPU
#   targets. Typically this is set on runtime components that are intended to
#    work for all supported targets, whereas it is ommitted for components which
#    are meant to be built only for the given targets (typically for kernel
#    libraries).
# DISABLE_AMDGPU_TARGETS: Do not set any GPU_TARGETS or AMDGPU_TARGETS variables
#   in the project. This is largely used for broken projects that cannot
#   build with an explicit target list.
# NO_MERGE_COMPILE_COMMANDS: Option to disable merging of this project's
#   compile_commands.json into the overall project. This is useful for
#   third-party projects that are excluded from all as it eliminates a
#   dependency that forces them to be downloaded/built.
# SOURCE_DIR: Absolute path to the external source directory.
# DIR_PREFIX: By default, directories named "build", "stage", "stamp" are
#   created. But if there are multiple sub-projects in a parent dir, then they
#   all must have a distinct prefix (not recommended).
# INSTALL_DESTINATION: Sub-directory within the stage/dist directory where this
#   sub-project installs. Defaults to empty, meaning that it installs at the top
#   of the namespace.
# CMAKE_ARGS: Additional CMake configure arguments.
# CMAKE_INCLUDES: Additional CMake files to include at the top level.
# BUILD_DEPS: Projects which must build and provide their packages prior to this
#   one.
# RUNTIME_DEPS: Projects which must build prior to this one and whose install
#   files must be distributed with this project's artifacts in order to
#   function.
# INTERFACE_INCLUDE_DIRS: Relative paths within the install tree which dependent
#   sub-projects must have their CPP include path set to include. Use of this
#   is strongly discouraged (deps should be on proper CMake libraries), but
#   some old projects still need this.
# INTERFACE_LINK_DIRS: Relative paths within the install tree which dependent
#   sub-projects must add to their runtime link library path.
# INTERFACE_PROGRAM_DIRS: Relative paths within the install tree which
#   dependent sub-projects must add to their program search path.
# INTERFACE_PKG_CONFIG_DIRS: Relative paths within the install tree which
#   dependent sub-projects must add to their PKG_CONFIG_PATH.
# IGNORE_PACKAGES: List of find_package package names to ignore, even if they
#   are advertised by the super-project. These packages will always fall-through
#   to the system resolver.
# COMPILER_TOOLCHAIN: Uses a built compiler toolchain instead of the
#   super-project specified C/C++ compiler. This will add an implicit dep on
#   the named compiler sub-project and reconfigure CMAKE_C(XX)_COMPILER options.
#   Only a fixed set of supported toolchains are supported (currently
#   "amd-llvm").
# BACKGROUND_BUILD: Option to indicate that the subproject does low concurrency,
#   high latency build steps. It will be run in the backgroun in a job pool that
#   allows some overlapping of work (controlled by
#   THEROCK_BACKGROUND_BUILD_JOBS).
# CMAKE_LISTS_RELPATH: Relative path within the source directory to the
#   CMakeLists.txt.
# EXTRA_DEPENDS: Extra target dependencies to add to the configure command.
# OUTPUT_ON_FAILURE: If given, build commands will produce no output unless if
#   it fails (logs will still be written). While generally not good to squelch a
#   "chatty" build, some third party libraries are hopeless and provide little
#   signal.
#
# RPATH handling:
# Each subproject has default logic injected which configures the INSTALL_RPATH
# of any defined executable or shared library targets. This default behavior can
# be disabled by setting NO_INSTALL_RPATH.
# *IMPORTANT:* This *overrides* any project local install RPATH configuration.
# While it may seem that this is heavy handed, in practice, almost no library
# on its own does the right thing. If it does, opt it out.
#
# In order to compute a correct RPATH (which we always set as relative to
# the installation prefix), it must be possible to determine each target's
# ORIGIN, as the OS will see it in the install tree. CMake does not have a
# way to determine this, so we use the following heuristic:
#   * Use THEROCK_INSTALL_RPATH_ORIGIN target property if defined.
#   * For executables, use the project level THEROCK_INSTALL_RPATH_EXECUTABLE_DIR.
#   * For shared libraries, use the project level THEROCK_INSTALL_RPATH_LIBRARY_DIR.
#
# We then get the computed private install RPATH dirs (which is the combination
# of any transitive INTERFACE_INSTALL_RPATH_DIRS) and transform them to an
# origin relative path and set them on the INSTALL_RPATH property.
#
# INSTALL_RPATH_DIRS: Install-tree relative paths to runtime shared library
#   dependencies to be used for targets defined directly in this project.
# INTERFACE_INSTALL_RPATH_DIRS: Like INSTALL_RPATH_DIRS but advertises these
#   directories to dependent projects (but does not use them itself).
# INSTALL_RPATH_EXECUTABLE_DIR: Default install-tree relative path to assume
#   that all executables are installed to. Defaults to INSTALL_DESTINATION/bin.
#   Can be overriden on a per target basis by setting
#   THEROCK_INSTALL_RPATH_EXECUTABLE_DIR.
# INSTALL_RPATH_LIBRARY_DIR: Default install-tree relative path to assume
#   that all shared libraries are installed to. Defaults to
#   INSTALL_DESTINATION/lib. Can be overriden on a per target basis by setting
#   THEROCK_INSTALL_RPATH_LIBRARY_DIR.
#
# Note that all transitive keywords (i.e. "INTERFACE_" prefixes) only consider
# transitive deps along their RUNTIME_DEPS edges, not BUILD_DEPS.
function(therock_cmake_subproject_declare target_name)
  cmake_parse_arguments(
    PARSE_ARGV 1 ARG
    "ACTIVATE;USE_DIST_AMDGPU_TAGETS;DISABLE_AMDGPU_TARGETS;EXCLUDE_FROM_ALL;BACKGROUND_BUILD;NO_MERGE_COMPILE_COMMANDS;OUTPUT_ON_FAILURE;NO_INSTALL_RPATH"
    "EXTERNAL_SOURCE_DIR;BINARY_DIR;DIR_PREFIX;INSTALL_DESTINATION;COMPILER_TOOLCHAIN;INTERFACE_PROGRAM_DIRS;CMAKE_LISTS_RELPATH;INTERFACE_PKG_CONFIG_DIRS;INSTALL_RPATH_EXECUTABLE_DIR;INSTALL_RPATH_LIBRARY_DIR"
    "BUILD_DEPS;RUNTIME_DEPS;CMAKE_ARGS;CMAKE_INCLUDES;INTERFACE_INCLUDE_DIRS;INTERFACE_LINK_DIRS;IGNORE_PACKAGES;EXTRA_DEPENDS;INSTALL_RPATH_DIRS;INTERFACE_INSTALL_RPATH_DIRS"
  )
  if(TARGET "${target_name}")
    message(FATAL_ERROR "Cannot declare subproject '${target_name}': a target with that name already exists")
  endif()

  cmake_path(IS_ABSOLUTE ARG_EXTERNAL_SOURCE_DIR _source_is_absolute)
  if(_source_is_absolute)
    if(NOT ARG_BINARY_DIR)
      # TODO: Swap these lines once moved.
      set(ARG_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
      # message(FATAL_ERROR "If specifying an absolute SOURCE_DIR, BINARY_DIR must be specified")
    endif()
  else()
    if(NOT ARG_BINARY_DIR)
      set(ARG_BINARY_DIR "${ARG_EXTERNAL_SOURCE_DIR}")
    endif()
    cmake_path(ABSOLUTE_PATH ARG_EXTERNAL_SOURCE_DIR)
    cmake_path(ABSOLUTE_PATH ARG_BINARY_DIR BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  set(_cmake_source_dir "${ARG_EXTERNAL_SOURCE_DIR}")
  if(ARG_CMAKE_LISTS_RELPATH)
    cmake_path(APPEND _cmake_source_dir "${ARG_CMAKE_LISTS_RELPATH}")
  endif()

  message(STATUS "Including subproject ${target_name} (from ${_cmake_source_dir})")
  add_custom_target("${target_name}" COMMENT "Top level target to build the ${target_name} sub-project")

  # Build directory.
  set(_binary_dir "${ARG_BINARY_DIR}/${ARG_DIR_PREFIX}build")
  make_directory("${_binary_dir}")

  # Stage directory.
  set(_stage_dir "${ARG_BINARY_DIR}/${ARG_DIR_PREFIX}stage")
  make_directory("${_stage_dir}")

  # Dist directory.
  set(_dist_dir "${ARG_BINARY_DIR}/${ARG_DIR_PREFIX}dist")
  make_directory("${_dist_dir}")

  # Stamp directory.
  set(_stamp_dir "${ARG_BINARY_DIR}/${ARG_DIR_PREFIX}stamp")
  make_directory("${_stamp_dir}")

  # Prefix directory.
  set(_prefix_dir "${ARG_BINARY_DIR}/${ARG_DIR_PREFIX}prefix")
  make_directory("${_prefix_dir}")

  # Collect LINK_DIRS and PROGRAM_DIRS from explicit args and RUNTIME_DEPS.
  _therock_cmake_subproject_collect_runtime_deps(
      _private_include_dirs _private_link_dirs _private_program_dirs _private_pkg_config_dirs _interface_install_rpath_dirs
      _transitive_runtime_deps
      _transitive_configure_depend_files
      ${ARG_RUNTIME_DEPS})

  # Include dirs
  set(_declared_include_dirs "${ARG_INTERFACE_INCLUDE_DIRS}")
  _therock_cmake_subproject_absolutize(_declared_include_dirs "${_stage_dir}")
  set(_interface_include_dirs ${_private_include_dirs} ${_declared_include_dirs})

  # Link dirs
  set(_declared_link_dirs "${ARG_INTERFACE_LINK_DIRS}")
  _therock_cmake_subproject_absolutize(_declared_link_dirs "${_stage_dir}")
  # The link dirs that we advertise combine interface link dirs of runtime deps
  # and any that we declared.
  set(_interface_link_dirs ${_private_link_dirs} ${_declared_link_dirs})

  # Program dirs
  # Collect program dirs from explicit args and RUNTIME_DEPS.
  set(_declared_program_dirs "${ARG_INTERFACE_PROGRAM_DIRS}")
  _therock_cmake_subproject_absolutize(_declared_program_dirs "${_dist_dir}")
  # The program dirs that we advertise combine interface program dirs of
  # runtime deps and any that we declared.
  set(_interface_program_dirs ${_private_program_dirs} ${_declared_program_dirs})

  # PkgConfig dirs
  set(_declared_pkg_config_dirs "${ARG_INTERFACE_PKG_CONFIG_DIRS}")
  _therock_cmake_subproject_absolutize(_declared_pkg_config_dirs "${_stage_dir}")
  set(_interface_pkg_config_dirs ${_private_pkg_config_dirs} ${_declared_pkg_config_dirs})

  # Install RPATH dirs.
  set(_interface_install_rpath_dirs ${_interface_install_rpath_dirs} ${ARG_INTERFACE_INSTALL_RPATH_DIRS})
  set(_private_install_rpath_dirs ${ARG_INSTALL_RPATH_DIRS} ${_interface_install_rpath_dirs})

  # Dedup transitives.
  list(REMOVE_DUPLICATES _private_include_dirs)
  list(REMOVE_DUPLICATES _interface_include_dirs)
  list(REMOVE_DUPLICATES _private_link_dirs)
  list(REMOVE_DUPLICATES _interface_link_dirs)
  list(REMOVE_DUPLICATES _private_program_dirs)
  list(REMOVE_DUPLICATES _interface_program_dirs)
  list(REMOVE_DUPLICATES _transitive_runtime_deps)
  list(REMOVE_DUPLICATES _transitive_configure_depend_files)
  list(REMOVE_DUPLICATES _private_pkg_config_dirs)
  list(REMOVE_DUPLICATES _interface_pkg_config_dirs)
  list(REMOVE_DUPLICATES _interface_install_rpath_dirs)
  list(REMOVE_DUPLICATES _private_install_rpath_dirs)

  # RPATH Executable and Library dir.
  if(NOT ARG_INSTALL_RPATH_EXECUTABLE_DIR)
    if(ARG_INSTALL_DESTINATION)
      set(ARG_INSTALL_RPATH_EXECUTABLE_DIR ARG_INSTALL_DESTINATION)
      cmake_path(APPEND ARG_INSTALL_RPATH_EXECUTABLE_DIR "bin")
    else()
      set(ARG_INSTALL_RPATH_EXECUTABLE_DIR "bin")
    endif()
  endif()
  if(NOT ARG_INSTALL_RPATH_LIBRARY_DIR)
    if(ARG_INSTALL_DESTINATION)
      set(ARG_INSTALL_RPATH_LIBRARY_DIR ARG_INSTALL_DESTINATION)
      cmake_path(APPEND ARG_INSTALL_RPATH_EXECUTABLE_DIR "lib")
    else()
      set(ARG_INSTALL_RPATH_LIBRARY_DIR "lib")
    endif()
  endif()

  # Build pool determination.
  set(_build_pool)
  if(ARG_BACKGROUND_BUILD)
    set(_build_pool "therock_background")
  endif()

  # GPU Targets.
  if(ARG_DISABLE_AMDGPU_TARGETS)
    set(_gpu_targets)
  elseif(ARG_USE_DIST_AMDGPU_TAGETS)
    set(_gpu_targets "${THEROCK_DIST_AMDGPU_TARGETS}")
  else()
    set(_gpu_targets "${THEROCK_AMDGPU_TARGETS}")
  endif()

  set_target_properties("${target_name}" PROPERTIES
    THEROCK_SUBPROJECT cmake
    THEROCK_BUILD_POOL "${_build_pool}"
    THEROCK_AMDGPU_TARGETS "${_gpu_targets}"
    THEROCK_DISABLE_AMDGPU_TARGETS "${ARG_DISABLE_AMDGPU_TARGETS}"
    THEROCK_EXCLUDE_FROM_ALL "${ARG_EXCLUDE_FROM_ALL}"
    THEROCK_NO_MERGE_COMPILE_COMMANDS "${ARG_NO_MERGE_COMPILE_COMMANDS}"
    THEROCK_EXTERNAL_SOURCE_DIR "${ARG_EXTERNAL_SOURCE_DIR}"
    THEROCK_BINARY_DIR "${_binary_dir}"
    THEROCK_DIST_DIR "${_dist_dir}"
    THEROCK_STAGE_DIR "${_stage_dir}"
    THEROCK_INSTALL_DESTINATION "${ARG_INSTALL_DESTINATION}"
    THEROCK_STAMP_DIR "${_stamp_dir}"
    THEROCK_PREFIX_DIR "${_prefix_dir}"
    THEROCK_CMAKE_SOURCE_DIR "${_cmake_source_dir}"
    THEROCK_CMAKE_PROJECT_INIT_FILE "${ARG_BINARY_DIR}/${ARG_BUILD_DIR}_init.cmake"
    THEROCK_CMAKE_PROJECT_TOOLCHAIN_FILE "${ARG_BINARY_DIR}/${ARG_BUILD_DIR}_toolchain.cmake"
    THEROCK_CMAKE_ARGS "${ARG_CMAKE_ARGS}"
    THEROCK_CMAKE_INCLUDES "${ARG_CMAKE_INCLUDES}"
    # Non-transitive build deps.
    THEROCK_BUILD_DEPS "${ARG_BUILD_DEPS}"
    # Transitive runtime deps.
    THEROCK_RUNTIME_DEPS "${_transitive_runtime_deps}"
    # Include dirs that this project compiles with.
    THEROCK_PRIVATE_INCLUDE_DIRS "${_private_include_dirs}"
    # Include dirs that are advertised to dependents.
    THEROCK_INTERFACE_INCLUDE_DIRS "${_interface_include_dirs}"
    # That this project compiles with.
    THEROCK_PRIVATE_LINK_DIRS "${_private_link_dirs}"
    # Link dirs that are advertised to dependents
    THEROCK_INTERFACE_LINK_DIRS "${_interface_link_dirs}"
    # PkgConfig directories that this project must use.
    THEROCK_PRIVATE_PKG_CONFIG_DIRS "${_private_pkg_config_dirs}"
    # Directories that sub-projects must add to their PKG_CONFIG_PATH.
    THEROCK_INTERFACE_PKG_CONFIG_DIRS "${_interface_pkg_config_dirs}"
    # Program dirs that this sub-project must configure with.
    THEROCK_PRIVATE_PROGRAM_DIRS "${_private_program_dirs}"
    # Program dirs that are advertised to dependents.
    THEROCK_INTERFACE_PROGRAM_DIRS "${_interface_program_dirs}"
    THEROCK_IGNORE_PACKAGES "${ARG_IGNORE_PACKAGES}"
    THEROCK_COMPILER_TOOLCHAIN "${ARG_COMPILER_TOOLCHAIN}"
    # Any extra depend files that must be added to the configure phase of dependents.
    THEROCK_INTERFACE_CONFIGURE_DEPEND_FILES "${_transitive_configure_depend_files}"
    THEROCK_EXTRA_DEPENDS "${ARG_EXTRA_DEPENDS}"
    THEROCK_OUTPUT_ON_FAILURE "${ARG_OUTPUT_ON_FAILURE}"

    # RPATH
    THEROCK_NO_INSTALL_RPATH "${ARG_NO_INSTALL_RPATH}"
    THEROCK_INTERFACE_INSTALL_RPATH_DIRS "${_interface_install_rpath_dirs}"
    THEROCK_PRIVATE_INSTALL_RPATH_DIRS "${_private_install_rpath_dirs}"
    THEROCK_INSTALL_RPATH_EXECUTABLE_DIR "${ARG_INSTALL_RPATH_EXECUTABLE_DIR}"
    THEROCK_INSTALL_RPATH_LIBRARY_DIR "${ARG_INSTALL_RPATH_LIBRARY_DIR}"
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
get_property(existing_packages GLOBAL PROPERTY THEROCK_ALL_PROVIDED_PACKAGES)
  if("${package_name}" IN_LIST existing_packages)
    message(SEND_ERROR "Duplicate package provided by ${target_name}: ${package_name}")
  endif()
  set_property(GLOBAL APPEND PROPERTY THEROCK_ALL_PROVIDED_PACKAGES "${package_name}")

  get_target_property(_existing_packages "${target_name}" THEROCK_PROVIDE_PACKAGES)
  if(${package_name} IN_LIST _existing_packages)
    message(FATAL_ERROR "Package defined multiple times on sub-project ${target_name}: ${package_name}")
  endif()
  set_property(TARGET "${target_name}" APPEND PROPERTY THEROCK_PROVIDE_PACKAGES "${package_name}")
  set(_relpath_name THEROCK_PACKAGE_RELPATH_${package_name})
  set_property(TARGET "${target_name}" PROPERTY ${_relpath_name} "${relative_path}")
  if(THEROCK_VERBOSE)
    message(STATUS "PROVIDE ${package_name} = ${relative_path} (from ${target_name})")
  endif()
endfunction()

# therock_cmake_subproject_activate
# If using multi-step setup (i.e. without 'ACTIVATE' on the declare), then this
# must be called once all configuration is complete.
function(therock_cmake_subproject_activate target_name)
  _therock_assert_is_cmake_subproject("${target_name}")

  # Get properties.
  get_target_property(_binary_dir "${target_name}" THEROCK_BINARY_DIR)
  get_target_property(_build_deps "${target_name}" THEROCK_BUILD_DEPS)
  get_target_property(_build_pool "${target_name}" THEROCK_BUILD_POOL)
  get_target_property(_compiler_toolchain "${target_name}" THEROCK_COMPILER_TOOLCHAIN)
  get_target_property(_transitive_configure_depend_files "${target_name}" THEROCK_INTERFACE_CONFIGURE_DEPEND_FILES)
  get_target_property(_dist_dir "${target_name}" THEROCK_DIST_DIR)
  get_target_property(_runtime_deps "${target_name}" THEROCK_RUNTIME_DEPS)
  get_target_property(_cmake_args "${target_name}" THEROCK_CMAKE_ARGS)
  get_target_property(_cmake_includes "${target_name}" THEROCK_CMAKE_INCLUDES)
  get_target_property(_cmake_project_init_file "${target_name}" THEROCK_CMAKE_PROJECT_INIT_FILE)
  get_target_property(_cmake_project_toolchain_file "${target_name}" THEROCK_CMAKE_PROJECT_TOOLCHAIN_FILE)
  get_target_property(_cmake_source_dir "${target_name}" THEROCK_CMAKE_SOURCE_DIR)
  get_target_property(_exclude_from_all "${target_name}" THEROCK_EXCLUDE_FROM_ALL)
  get_target_property(_external_source_dir "${target_name}" THEROCK_EXTERNAL_SOURCE_DIR)
  get_target_property(_extra_depends "${target_name}" THEROCK_EXTRA_DEPENDS)
  get_target_property(_ignore_packages "${target_name}" THEROCK_IGNORE_PACKAGES)
  get_target_property(_install_destination "${target_name}" THEROCK_INSTALL_DESTINATION)
  get_target_property(_no_merge_compile_commands "${target_name}" THEROCK_NO_MERGE_COMPILE_COMMANDS)
  get_target_property(_private_include_dirs "${target_name}" THEROCK_PRIVATE_INCLUDE_DIRS)
  get_target_property(_private_link_dirs "${target_name}" THEROCK_PRIVATE_LINK_DIRS)
  get_target_property(_private_pkg_config_dirs "${target_name}" THEROCK_PRIVATE_PKG_CONFIG_DIRS)
  get_target_property(_private_program_dirs "${target_name}" THEROCK_PRIVATE_PROGRAM_DIRS)
  get_target_property(_stage_dir "${target_name}" THEROCK_STAGE_DIR)
  get_target_property(_sources "${target_name}" SOURCES)
  get_target_property(_stamp_dir "${target_name}" THEROCK_STAMP_DIR)
  get_target_property(_prefix_dir "${target_name}" THEROCK_PREFIX_DIR)
  get_target_property(_output_on_failure "${target_name}" THEROCK_OUTPUT_ON_FAILURE)
  # RPATH properties: just mirror these to same named variables because we just
  # mirror them syntactically into the subprojet..
  get_target_property(THEROCK_NO_INSTALL_RPATH "${target_name}" THEROCK_NO_INSTALL_RPATH)
  get_target_property(THEROCK_INTERFACE_INSTALL_RPATH_DIRS "${target_name}" THEROCK_INTERFACE_INSTALL_RPATH_DIRS)
  get_target_property(THEROCK_PRIVATE_INSTALL_RPATH_DIRS "${target_name}" THEROCK_PRIVATE_INSTALL_RPATH_DIRS)
  get_target_property(THEROCK_INSTALL_RPATH_EXECUTABLE_DIR "${target_name}" THEROCK_INSTALL_RPATH_EXECUTABLE_DIR)
  get_target_property(THEROCK_INSTALL_RPATH_LIBRARY_DIR "${target_name}" THEROCK_INSTALL_RPATH_LIBRARY_DIR)


  # Handle optional properties.
  if(NOT _sources)
    set(_sources)
  endif()

  # Defaults.
  set(_configure_comment_suffix)
  set(_build_comment_suffix)

  # Detect pre/post hooks.
  set(_pre_hook_path "${CMAKE_CURRENT_SOURCE_DIR}/pre_hook_${target_name}.cmake")
  if(NOT EXISTS "${_pre_hook_path}")
    set(_pre_hook_path)
  endif()
  set(_post_hook_path "${CMAKE_CURRENT_SOURCE_DIR}/post_hook_${target_name}.cmake")
  if(NOT EXISTS "${_post_hook_path}")
    set(_post_hook_path)
  endif()

  # Report transitive runtime deps.
  if(_runtime_deps AND THEROCK_VERBOSE)
    list(JOIN _runtime_deps " " _runtime_deps_pretty)
    message(STATUS "  RUNTIME_DEPS: ${_runtime_deps_pretty}")
  endif()

  get_property(_mirror_cmake_vars GLOBAL PROPERTY THEROCK_DEFAULT_CMAKE_VARS)

  # Pairs of arguments for a `cmake -E env` command to run before each
  # subproject build or configure command.
  # https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-E-arg-env
  # TODO: split into 'build' and 'configure'? Keeping them in sync seems useful.
  set(_build_env_pairs)

  # All project dependencies are managed within the super-project so we don't
  # want subprojects reaching outside of the sandbox and building against
  # uncontrolled (and likely incompatible) sources.
  #
  # These environment variables have been used by some subprojects to discover
  # preexisting ROCm/HIP SDK installs. If detected, these subprojects then do
  # things like:
  #   * Append `${HIP_PATH}/cmake` to `CMAKE_MODULE_PATH`
  #   * Use `${HIP_PATH}` as a hint for `find_package()` calls
  # We unset both the CMake and environment variables with these names.
  # See also https://github.com/ROCm/TheRock/issues/670.
  list(APPEND _build_env_pairs "--unset=ROCM_PATH")
  list(APPEND _build_env_pairs "--unset=ROCM_DIR")
  list(APPEND _build_env_pairs "--unset=HIP_PATH")
  list(APPEND _build_env_pairs "--unset=HIP_DIR")

  # Handle compiler toolchain.
  set(_compiler_toolchain_addl_depends)
  set(_compiler_toolchain_init_contents)
  _therock_cmake_subproject_setup_toolchain("${target_name}"
    "${_compiler_toolchain}" "${_cmake_project_toolchain_file}")

  # Customize any other super-project CMake variables that are captured by
  # _init.cmake.
  if(_private_program_dirs)
    set(CMAKE_PROGRAM_PATH ${_private_program_dirs} ${CMAKE_PROGRAM_PATH})
    if(THEROCK_VERBOSE)
      foreach(_message_contents ${_private_program_dirs})
        message(STATUS "  PROGRAM_DIR: ${_message_contents}")
      endforeach()
    endif()
  endif()

  # Generate the project_init.cmake
  set(_dep_provider_file)
  if(_build_deps OR _runtime_deps)
    set(_dep_provider_file "${THEROCK_SOURCE_DIR}/cmake/therock_subproject_dep_provider.cmake")
  endif()

  set(_init_contents)
  # Support generator expressions in install CODE
  # We rely on this for debug symbol separation and some of our very old projects
  # have a CMake minver < 3.14, defaulting them to OLD. Unfortunately, this policy
  # must be set globally vs in a block scope to have effect.
  string(APPEND _init_contents "cmake_policy(SET CMP0087 NEW)\n")
  foreach(_var_name ${_mirror_cmake_vars})
    string(APPEND _init_contents "set(${_var_name} \"@${_var_name}@\" CACHE STRING \"\" FORCE)\n")
  endforeach()
  # Process dependencies. We process runtime deps first so that they take precedence
  # over build deps (first wins). Both come from the dist directory because if
  # build tools are needed from them, only the dist dir is guaranteed to have
  # all runtime deps met.
  string(APPEND _init_contents "set(THEROCK_PROVIDED_PACKAGES)\n")
  set(_deps_contents)
  set(_deps_provided)
  _therock_cmake_subproject_setup_deps(_deps_contents _deps_provided THEROCK_DIST_DIR ${_runtime_deps})
  _therock_cmake_subproject_setup_deps(_deps_contents _deps_provided THEROCK_DIST_DIR ${_build_deps})

  string(APPEND _init_contents "${_deps_contents}")
  string(APPEND _init_contents "set(THEROCK_IGNORE_PACKAGES \"@_ignore_packages@\")\n")
  string(APPEND _init_contents "list(PREPEND CMAKE_MODULE_PATH \"${THEROCK_SOURCE_DIR}/cmake/finders\")\n")
  string(APPEND _init_contents "list(PREPEND CMAKE_PREFIX_PATH \"@_prefix_dir@\")\n")
  get_property(_all_provided_packages GLOBAL PROPERTY THEROCK_ALL_PROVIDED_PACKAGES)
  string(APPEND _init_contents "set(THEROCK_STRICT_PROVIDED_PACKAGES \"@_all_provided_packages@\")\n")

  # Include dirs.
  foreach(_private_include_dir ${_private_include_dirs})
    if(THEROCK_VERBOSE)
      message(STATUS "  INCLUDE_DIR: ${_private_include_dir}")
    endif()
    string(APPEND _init_contents "include_directories(BEFORE \"${_private_include_dir}\")\n")
    string(APPEND _init_contents "list(PREPEND CMAKE_REQUIRED_INCLUDES \"${_private_include_dir}\")\n")
    string(APPEND _init_contents "list(APPEND THEROCK_SUPERPROJECT_INCLUDE_DIRS \"${_private_include_dir}\")\n")
  endforeach()

  # Link dirs.
  foreach(_private_link_dir ${_private_link_dirs})
    if(THEROCK_VERBOSE)
      message(STATUS "  LINK_DIR: ${_private_link_dir}")
    endif()
    # Make the link dir visible to CMake find_library.
    string(APPEND _init_contents "list(APPEND CMAKE_LIBRARY_PATH \"${_private_link_dir}\")\n")
    if(NOT MSVC)
      # The normal way.
      string(APPEND _init_contents "string(APPEND CMAKE_EXE_LINKER_FLAGS \" -L ${_private_link_dir} -Wl,-rpath-link,${_private_link_dir}\")\n")
      string(APPEND _init_contents "string(APPEND CMAKE_SHARED_LINKER_FLAGS \" -L ${_private_link_dir} -Wl,-rpath-link,${_private_link_dir}\")\n")
    elseif(_compiler_toolchain STREQUAL "amd-llvm" OR _compiler_toolchain STREQUAL "amd-hip")
      # The Windows but using a clang-based toolchain way.
      #   Working around "lld-link: warning: ignoring unknown argument '-rpath-link'"
      string(APPEND _init_contents "string(APPEND CMAKE_EXE_LINKER_FLAGS \" -L ${_private_link_dir} \")\n")
      string(APPEND _init_contents "string(APPEND CMAKE_SHARED_LINKER_FLAGS \" -L ${_private_link_dir} \")\n")
    else()
      # The MSVC way.
      string(APPEND _init_contents "string(APPEND CMAKE_EXE_LINKER_FLAGS \" /LIBPATH:${_private_link_dir}\")\n")
      string(APPEND _init_contents "string(APPEND CMAKE_SHARED_LINKER_FLAGS \" /LIBPATH:${_private_link_dir}\")\n")
    endif()
  endforeach()

  if(THEROCK_VERBOSE AND _private_pkg_config_dirs)
    message(STATUS "  PKG_CONFIG_DIRS: ${_private_pkg_config_dirs}")
  endif()
  string(APPEND _init_contents "set(THEROCK_PKG_CONFIG_DIRS \"@_private_pkg_config_dirs@\")\n")

  string(APPEND _init_contents "${_compiler_toolchain_init_contents}")
  if(_dep_provider_file)
    string(APPEND _init_contents "include(${_dep_provider_file})\n")
  endif()
  if(_pre_hook_path)
    string(APPEND _init_contents "include(@_pre_hook_path@)\n")
  endif()
  string(APPEND _init_contents "set(THEROCK_USER_POST_HOOK)\n")
  if(_post_hook_path)
    string(APPEND _init_contents "set(THEROCK_USER_POST_HOOK \"@_post_hook_path@\")\n")
  endif()
  set(_global_post_include "${THEROCK_SOURCE_DIR}/cmake/therock_global_post_subproject.cmake")
  string(APPEND _init_contents "cmake_language(DEFER CALL include \"@_global_post_include@\")\n")
  foreach(_addl_cmake_include ${_cmake_includes})
    if(NOT IS_ABSOLUTE)
      find_path(_addl_cmake_include_path "${addl_cmake_include}" NO_CACHE NO_DEFAULT_PATH PATHS ${CMAKE_MODULE_PATH} REQUIRED)
      cmake_path(ABSOLUTE_PATH _addl_cmake_include BASE_DIRECTORY "${_addl_cmake_include_path}")
    endif()
    string(APPEND _init_contents "include(\"${_addl_cmake_include}\")\n")
  endforeach()
  file(CONFIGURE OUTPUT "${_cmake_project_init_file}" CONTENT "${_init_contents}" @ONLY ESCAPE_QUOTES)

  # Transform build and run deps from target form (i.e. 'ROCR-Runtime' to a dependency
  # on the dist.stamp file). These are a dependency for configure. We satisfy both
  # build and runtime deps from the dist phase because even build-only deps may
  # need to execute tools linked such that they require all transitive libraries
  # materialized. We might be able to save some milliseconds by steering
  # build-only deps to the stage.stamp file, but the complexity involved is not
  # worth it, especially considering that it increases the likelihood of build
  # non-determinism.
  set(_configure_dep_stamps)
  _therock_cmake_subproject_deps_to_stamp(_configure_dep_stamps dist.stamp ${_build_deps})
  _therock_cmake_subproject_deps_to_stamp(_configure_dep_stamps dist.stamp ${_runtime_deps})

  # Target flags.
  set(_all_option)
  if(NOT _exclude_from_all)
    set(_all_option "ALL")
  endif()

  # Detect whether the stage dir has been pre-built.
  set(_prebuilt_file "${_stage_dir}.prebuilt")
  set(_configure_stamp_file "${_stamp_dir}/configure.stamp")
  set(_build_stamp_file "${_stamp_dir}/build.stamp")
  set(_stage_stamp_file "${_stamp_dir}/stage.stamp")

  # Derive the CMAKE_BUILD_TYPE from eiether {project}_BUILD_TYPE or the global
  # CMAKE_BUILD_TYPE.
  set(_cmake_build_type "${${target_name}_BUILD_TYPE}")
  if(NOT _cmake_build_type)
    set(_cmake_build_type "${CMAKE_BUILD_TYPE}")
  else()
    message(STATUS "  PROJECT SPECIFIC CMAKE_BUILD_TYPE=${_cmake_build_type}")
  endif()

  if(EXISTS "${_prebuilt_file}")
    # If pre-built, just touch the stamp files, conditioned on the prebuilt
    # marker file (which may just be a stamp file or may contain a unique hash
    # for this part of the build).
    message(STATUS "  DISABLING BUILD: Marked as pre-built")
    add_custom_command(
      OUTPUT "${_configure_stamp_file}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_configure_stamp_file}"
      DEPENDS "${_prebuilt_file}"
    )
    add_custom_command(
      OUTPUT "${_build_stamp_file}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_build_stamp_file}"
      DEPENDS "${_prebuilt_file}"
    )
    add_custom_command(
      OUTPUT "${_stage_stamp_file}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_stage_stamp_file}"
      DEPENDS "${_prebuilt_file}"
    )
  else()
    # Not pre-built: normal configure/build/stage install.
    # configure target
    if(THEROCK_VERBOSE)
      message(STATUS "  CONFIGURE_DEPENDS: ${_transitive_configure_depend_files} ")
    endif()
    set(_configure_comment_suffix " (in background)")
    set(_terminal_option)
    set(_build_terminal_option "USES_TERMINAL")
    if("$ENV{THEROCK_INTERACTIVE}")
      set(_terminal_option "USES_TERMINAL")
      set(_configure_comment_suffix)
    elseif(_build_pool)
      if(THEROCK_VERBOSE)
        message(STATUS "  JOB_POOL: ${_build_pool}")
      endif()
      set(_build_terminal_option JOB_POOL "${_build_pool}")
      set(_build_comment_suffix " (in background)")
    endif()
    set(_stage_destination_dir "${_stage_dir}")
    if(_install_destination)
      cmake_path(APPEND _stage_destination_dir "${_install_destination}")
    endif()
    set(_compile_commands_file "${PROJECT_BINARY_DIR}/compile_commands_fragment_${target_name}.json")
    therock_subproject_log_command(_configure_log_prefix
      LOG_FILE "${target_name}_configure.log"
      LABEL "${target_name} configure"
      OUTPUT_ON_FAILURE "${_output_on_failure}"
    )
    add_custom_command(
      OUTPUT "${_configure_stamp_file}"
      COMMAND
        ${_configure_log_prefix}
        "${CMAKE_COMMAND}" -E env ${_build_env_pairs} --
        "${CMAKE_COMMAND}"
        "-G${CMAKE_GENERATOR}"
        "-B${_binary_dir}"
        "-S${_cmake_source_dir}"
        "-DCMAKE_BUILD_TYPE=${_cmake_build_type}"
        "-DCMAKE_INSTALL_PREFIX=${_stage_destination_dir}"
        "-DTHEROCK_STAGE_INSTALL_ROOT=${_stage_dir}"
        "-DCMAKE_TOOLCHAIN_FILE=${_cmake_project_toolchain_file}"
        "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${_cmake_project_init_file}"
        ${_cmake_args}
      # CMake doesn't always generate a compile_commands.json so touch one to keep
      # the build graph sane.
      COMMAND "${CMAKE_COMMAND}" -E touch "${_binary_dir}/compile_commands.json"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_configure_stamp_file}"
      COMMAND "${CMAKE_COMMAND}" -E copy "${_binary_dir}/compile_commands.json" "${_compile_commands_file}"
      WORKING_DIRECTORY "${_binary_dir}"
      COMMENT "Configure sub-project ${target_name}${_configure_comment_suffix}"
      ${_terminal_option}
      BYPRODUCTS
        "${_binary_dir}/CMakeCache.txt"
        "${_binary_dir}/cmake_install.cmake"
        "${_compile_commands_file}"
      DEPENDS
        "${_cmake_source_dir}/CMakeLists.txt"
        "${_cmake_project_toolchain_file}"
        "${_cmake_project_init_file}"
        "${_global_post_include}"
        ${_extra_depends}
        ${_dep_provider_file}
        ${_configure_dep_stamps}
        ${_pre_hook_path}
        ${_post_hook_path}
        ${_compiler_toolchain_addl_depends}
        ${_transitive_configure_depend_files}

        # TODO: Have a mechanism for adding more depends for better rebuild ergonomics
    )
    add_custom_target(
      "${target_name}+configure"
      ${_all_option}
      DEPENDS "${_configure_stamp_file}"
    )
    add_dependencies("${target_name}" "${target_name}+configure")
    if(NOT _no_merge_compile_commands)
      set_property(GLOBAL APPEND PROPERTY THEROCK_SUBPROJECT_COMPILE_COMMANDS_FILES
        "${_compile_commands_file}"
      )
    endif()

    # build target.
    therock_subproject_log_command(_build_log_prefix
      LOG_FILE "${target_name}_build.log"
      LABEL "${target_name}"
      OUTPUT_ON_FAILURE "${_output_on_failure}"
    )
    add_custom_command(
      OUTPUT "${_build_stamp_file}"
      COMMAND
        ${_build_log_prefix}
        "${CMAKE_COMMAND}" -E env ${_build_env_pairs} --
        "${CMAKE_COMMAND}" "--build" "${_binary_dir}"
      COMMAND "${CMAKE_COMMAND}" -E touch "${_build_stamp_file}"
      WORKING_DIRECTORY "${_binary_dir}"
      COMMENT "Building sub-project ${target_name}${_build_comment_suffix}"
      ${_build_terminal_option}
      DEPENDS
        "${_configure_stamp_file}"
        ${_sources}
    )
    add_custom_target(
      "${target_name}+build"
      ${_all_option}
      DEPENDS
        "${_build_stamp_file}"
    )
    add_dependencies("${target_name}" "${target_name}+build")

    # stage install target.
    set(_install_strip_option)
    if(THEROCK_SPLIT_DEBUG_INFO)
      set(_install_strip_option "--strip")
    endif()
    therock_subproject_log_command(_install_log_prefix
      LOG_FILE "${target_name}_install.log"
      LABEL "${target_name} install"
      # While useful for debugging, stage install logs are almost pure noise
      # for interactive use.
      OUTPUT_ON_FAILURE "${THEROCK_QUIET_INSTALL}"
    )
    add_custom_command(
      OUTPUT "${_stage_stamp_file}"
      COMMAND ${_install_log_prefix} "${CMAKE_COMMAND}" --install "${_binary_dir}" ${_install_strip_option}
      COMMAND "${CMAKE_COMMAND}" -E touch "${_stage_stamp_file}"
      WORKING_DIRECTORY "${_binary_dir}"
      COMMENT "Stage installing sub-project ${target_name}"
      ${_terminal_option}
      DEPENDS
        "${_build_stamp_file}"
    )
    add_custom_target(
      "${target_name}+stage"
      ${_all_option}
      DEPENDS
        "${_stage_stamp_file}"
    )
    add_dependencies("${target_name}" "${target_name}+stage")
  endif()

  # dist install target.
  set(_dist_stamp_file "${_stamp_dir}/dist.stamp")
  set(_fileset_tool "${THEROCK_SOURCE_DIR}/build_tools/fileset_tool.py")
  _therock_cmake_subproject_get_stage_dirs(
    _dist_source_dirs "${target_name}" ${_runtime_deps})
  add_custom_command(
    OUTPUT "${_dist_stamp_file}"
    COMMAND "${Python3_EXECUTABLE}" "${_fileset_tool}" copy "${_dist_dir}" ${_dist_source_dirs}
    COMMAND "${CMAKE_COMMAND}" -E touch "${_dist_stamp_file}"
    COMMENT "Merging sub-project dist directory for ${target_name}"
    ${_terminal_option}
    DEPENDS
      "${_stage_stamp_file}"
      "${_fileset_tool}"
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
      ${CMAKE_COMMAND} -E rm -rf "${_binary_dir}" "${_stage_dir}" "${_stamp_dir}" "${_dist_dir}"
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

# Gets the dist/ directory for a sub-project.
function(therock_cmake_subproject_dist_dir out_var target_name)
  _therock_assert_is_cmake_subproject("${target_name}")
  get_target_property(_dir "${target_name}" THEROCK_DIST_DIR)
  set("${out_var}" "${_dir}" PARENT_SCOPE)
endfunction()

# Merges all compile_commands.json files and exports them.
function(therock_subproject_merge_compile_commands)
  if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    return()
  endif()

  message(STATUS "Setting up compile_commands.json export")
  get_property(_fragment_files GLOBAL PROPERTY THEROCK_SUBPROJECT_COMPILE_COMMANDS_FILES)
  if(EXISTS "${CMAKE_BINARY_DIR}/compile_commands.json")
    list(APPEND _fragment_files "${CMAKE_BINARY_DIR}/compile_commands.json")
  endif()

  set(_merged_file "${PROJECT_BINARY_DIR}/compile_commands_merged.json")
  set(_merged_file_in_source_dir "${PROJECT_SOURCE_DIR}/compile_commands.json")

  set(_merge_script "${THEROCK_SOURCE_DIR}/build_tools/merge_compile_commands.py")
  add_custom_command(
    OUTPUT "${_merged_file}"
    COMMENT "Merging compile_commands.json"
    COMMAND "${Python3_EXECUTABLE}"
      "${_merge_script}" "${_merged_file}" ${_fragment_files}
    COMMAND "${CMAKE_COMMAND}" -E copy "${_merged_file}" "${_merged_file_in_source_dir}"
    BYPRODUCTS
       "${_merged_file_in_source_dir}"
    DEPENDS
      "${_merge_script}"
      ${_fragment_files}
  )
  add_custom_target(therock_merged_compile_commands ALL
    DEPENDS
      "${_merged_file}"
      "${_merge_script}"
      ${_fragment_files}
  )
endfunction()

function(_therock_assert_is_cmake_subproject target_name)
  # Make sure it is a sub-project.
  get_target_property(_is_subproject "${target_name}" THEROCK_SUBPROJECT)
  if(NOT _is_subproject STREQUAL "cmake")
    message(FATAL_ERROR "Target ${target_name} is not a sub-project")
  endif()
endfunction()

# Builds a CMake language fragment to set up a dependency provider such that
# it handles super-project provided dependencies locally.
function(_therock_cmake_subproject_setup_deps out_contents out_provided dep_dir_property)
  string(APPEND CMAKE_MESSAGE_INDENT "  ")
  set(_contents "${${out_contents}}")
  set(_already_provided ${${out_provided}})
  foreach(dep_target ${ARGN})
    _therock_assert_is_cmake_subproject("${dep_target}")

    get_target_property(_provides "${dep_target}" THEROCK_PROVIDE_PACKAGES)
    if(_provides)
      foreach(_package_name ${_provides})
        if(_package_name IN_LIST _already_provided)
          continue()
        endif()
        list(APPEND _already_provided "${_package_name}")
        get_target_property(_dep_dir "${dep_target}" "${dep_dir_property}")
        set(_relpath_name THEROCK_PACKAGE_RELPATH_${_package_name})
        get_target_property(_relpath "${dep_target}" ${_relpath_name})
        if(NOT _dep_dir OR NOT _relpath)
          message(FATAL_ERROR "Missing package info props for ${_package_name} on ${dep_target}: '${_dep_dir}' ${_relpath_name}='${_relpath}'")
        endif()
        set(_find_package_path "${_dep_dir}")
        cmake_path(APPEND _find_package_path "${_relpath}")
        if(THEROCK_VERBOSE)
          message(STATUS "INJECT ${_package_name} = ${_find_package_path} (from ${dep_target})")
        endif()
        string(APPEND _contents "set(THEROCK_PACKAGE_DIR_${_package_name} \"${_find_package_path}\")\n")
        string(APPEND _contents "list(APPEND THEROCK_PROVIDED_PACKAGES ${_package_name})\n")

        # Now write a trampoline package config file into the prefix so that normal
        # prefix based find_package() will resolve properly.
        string(TOLOWER "${_package_name}" _package_name_lower)
        set(_config_file "${_prefix_dir}/${_package_name}Config.cmake")
        configure_file(
          "${THEROCK_SOURCE_DIR}/cmake/templates/find_config.tmpl.cmake"
          "${_config_file}" @ONLY
        )
        set(_version_file "${_prefix_dir}/${_package_name}ConfigVersion.cmake")
        configure_file(
          "${THEROCK_SOURCE_DIR}/cmake/templates/find_config_version.tmpl.cmake"
          "${_version_file}" @ONLY
        )
      endforeach()
    endif()
  endforeach()
  set("${out_contents}" "${_contents}" PARENT_SCOPE)
  set("${out_provided}" "${_already_provided}" PARENT_SCOPE)
endfunction()

# Gets the staging install directories for a list of subproject deps.
function(_therock_cmake_subproject_get_stage_dirs out_dirs)
  set(_dirs)
  foreach(target_name ${ARGN})
    get_target_property(_stage_dir "${target_name}" THEROCK_STAGE_DIR)
    if(NOT _stage_dir)
      message(FATAL_ERROR "Sub-project target ${target_name} does not have a stage install dir")
    endif()
    list(APPEND _dirs "${_stage_dir}")
  endforeach()
  set(${out_dirs} "${_dirs}" PARENT_SCOPE)
endfunction()

# Transforms a list of sub-project targets to corresponding stamp files of
# `stamp_name`. These are the actual build system deps that are encoded in the
# commands (whereas the target names are just for humans).
function(_therock_cmake_subproject_deps_to_stamp out_stamp_files stamp_name)
  set(_stamp_files ${${out_stamp_files}})
  foreach(target_name ${ARGN})
    _therock_assert_is_cmake_subproject("${target_name}")
    get_target_property(_stamp_dir "${target_name}" THEROCK_STAMP_DIR)
    if(NOT _stamp_dir)
      message(FATAL_ERROR "Sub-project is missing a stamp dir: ${target_name}")
    endif()

    list(APPEND _stamp_files "${_stamp_dir}/${stamp_name}")
  endforeach()
  set(${out_stamp_files} "${_stamp_files}" PARENT_SCOPE)
endfunction()

# For a list of targets, gets absolute paths for all interface link directories
# and transitive runtime deps. Both lists may contain duplicates if the DAG
# includes the same dep multiple times.
function(_therock_cmake_subproject_collect_runtime_deps
    out_include_dirs out_link_dirs out_program_dirs out_pkg_config_dirs out_install_rpath_dirs
    out_transitive_deps
    out_transitive_configure_depend_files)
  set(_include_dirs)
  set(_install_rpath_dirs)
  set(_link_dirs)
  set(_program_dirs)
  set(_pkg_config_dirs)
  set(_transitive_deps)
  set(_transitive_configure_depend_files)
  foreach(target_name ${ARGN})
    _therock_assert_is_cmake_subproject("${target_name}")
    get_target_property(_declared_configure_depend_files "${target_name}" THEROCK_INTERFACE_CONFIGURE_DEPEND_FILES)
    list(APPEND _transitive_configure_depend_files ${_declared_configure_depend_files})
    get_target_property(_stamp_dir "${target_name}" THEROCK_STAMP_DIR)

    # Include dirs.
    get_target_property(_include_dir "${target_name}" THEROCK_INTERFACE_INCLUDE_DIRS)
    list(APPEND _include_dirs ${_include_dir})

    # Link dirs.
    get_target_property(_link_dir "${target_name}" THEROCK_INTERFACE_LINK_DIRS)
    list(APPEND _link_dirs ${_link_dir})

    # Transitive runtime target deps.
    get_target_property(_deps "${target_name}" THEROCK_RUNTIME_DEPS)
    list(APPEND _transitive_deps ${_deps} ${target_name})

    # If we have program dirs, then this target's 'dist' phase has to become
    # a transitive dep for all future configures (by default the build graph
    # only depends on the 'stage' phase).
    get_target_property(_program_dir "${target_name}" THEROCK_INTERFACE_PROGRAM_DIRS)
    if(_program_dir)
      list(APPEND _program_dirs ${_program_dir})
      list(APPEND _transitive_configure_depend_files "${_stamp_dir}/dist.stamp")
    endif()

    # PkgConfig dirs.
    get_target_property(_pkg_config_dir "${target_name}" THEROCK_INTERFACE_PKG_CONFIG_DIRS)
    if(_pkg_config_dir)
      list(APPEND _pkg_config_dirs ${_pkg_config_dir})
    endif()

    # RPATH dirs.
    get_target_property(_install_rpath_dir "${target_name}" THEROCK_INTERFACE_INSTALL_RPATH_DIRS)
    if(_install_rpath_dir)
      list(APPEND _install_rpath_dirs ${_install_rpath_dir})
    endif()
  endforeach()
  set("${out_include_dirs}" "${_include_dirs}" PARENT_SCOPE)
  set("${out_install_rpath_dirs}" "${_install_rpath_dirs}" PARENT_SCOPE)
  set("${out_link_dirs}" "${_link_dirs}" PARENT_SCOPE)
  set("${out_program_dirs}" "${_program_dirs}" PARENT_SCOPE)
  set("${out_pkg_config_dirs}" "${_pkg_config_dirs}" PARENT_SCOPE)
  set("${out_transitive_deps}" "${_transitive_deps}" PARENT_SCOPE)
  set("${out_transitive_configure_depend_files}" "${_transitive_configure_depend_files}" PARENT_SCOPE)
endfunction()

# Transforms a list to be absolute paths if not already.
function(_therock_cmake_subproject_absolutize list_var relative_to)
  set(_dirs "${${list_var}}")
  set(_abs_dirs)
  foreach(_dir ${_dirs})
    cmake_path(ABSOLUTE_PATH _dir BASE_DIRECTORY "${relative_to}" NORMALIZE)
    list(APPEND _abs_dirs "${_dir}")
  endforeach()
  set("${list_var}" "${_abs_dirs}" PARENT_SCOPE)
endfunction()

# Filters the target's THEROCK_AMDGPU_TARGETS property based on global settings for the project.
function(_therock_filter_project_gpu_targets out_var target_name)
  get_property(_excludes GLOBAL PROPERTY "THEROCK_AMDGPU_PROJECT_TARGET_EXCLUDES_${target_name}")
  get_target_property(_gpu_targets "${target_name}" THEROCK_AMDGPU_TARGETS)
  set(_filtered ${_gpu_targets})
  if(_excludes)
    foreach(exclude in ${_excludes})
      if("${exclude}" IN_LIST _filtered)
        message(WARNING
          "Excluding support for ${exclude} in ${target_name} because it was "
          "manually marked for exclusion in therock_amdgpu_targets.cmake. This "
          "warning should never be issued for production/supported gfx targets.")
        list(REMOVE_ITEM _filtered "${exclude}")
      endif()
    endforeach()
  endif()

  if(NOT _filtered)
    if("${target_name}" IN_LIST THEROCK_SUBPROJECTS_REQUIRING_DEFAULT_GPU_TARGETS)
      set(_filtered ${THEROCK_DEFAULT_GPU_TARGETS})
      message(WARNING
        "Project ${target_name} cannot build with no gpu targets but was "
        "instructed to do so. Overriding to the default ${_filtered}. "
        "This message should never appear for production/supported gfx targets."
      )
    endif()
  endif()

  set("${out_var}" "${_filtered}" PARENT_SCOPE)
endfunction()

# Writes a toolchain file and sets variables in the parent scope of the
# therock_cmake_subproject_activate prior to initializing sub-project arguments
# to configure the toolchain based on the user-provided COMPILER_TOOLCHAIN.
#
# Toolchain menemonics:
#   * amd-llvm: Locally build compiler/amd-llvm toolchain as a standalone
#     tool. While this can compiler HIP code, it does not natively have access
#     to a ROCM installation for headers, etc.
#   * amd-hip: Extends the amd-llvm toolchain to also depend on HIP, making
#     it ready to use to compile HIP code.
function(_therock_cmake_subproject_setup_toolchain
    target_name compiler_toolchain toolchain_file)
  string(APPEND CMAKE_MESSAGE_INDENT "  ")
  set(_build_env_pairs "${_build_env_pairs}")
  set(_toolchain_contents)

  get_target_property(_disable_amdgpu_targets "${target_name}" THEROCK_DISABLE_AMDGPU_TARGETS)
  set(_filtered_gpu_targets)
  if(NOT _disable_amdgpu_targets)
    _therock_filter_project_gpu_targets(_filtered_gpu_targets "${target_name}")
    # TODO: AMDGPU_TARGETS is being deprecated. For now we set both.
    string(APPEND _toolchain_contents "set(AMDGPU_TARGETS @_filtered_gpu_targets@ CACHE STRING \"From super-project\" FORCE)\n")
    string(APPEND _toolchain_contents "set(GPU_TARGETS @_filtered_gpu_targets@ CACHE STRING \"From super-project\" FORCE)\n")
  endif()

  # General settings applicable to all toolchains.
  string(APPEND _toolchain_contents "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n")
  string(APPEND _toolchain_contents "set(CMAKE_INSTALL_LIBDIR @CMAKE_INSTALL_LIBDIR@)\n")
  string(APPEND _toolchain_contents "set(CMAKE_PLATFORM_NO_VERSIONED_SONAME @CMAKE_PLATFORM_NO_VERSIONED_SONAME@)\n")

  # Propagate super-project flags to the sub-project by default.
  string(APPEND _toolchain_contents "set(CMAKE_C_COMPILER \"@CMAKE_C_COMPILER@\")\n")
  string(APPEND _toolchain_contents "set(CMAKE_CXX_COMPILER \"@CMAKE_CXX_COMPILER@\")\n")
  string(APPEND _toolchain_contents "set(CMAKE_LINKER \"@CMAKE_LINKER@\")\n")
  string(APPEND _toolchain_contents "set(CMAKE_C_COMPILER_LAUNCHER \"@CMAKE_C_COMPILER_LAUNCHER@\")\n")
  string(APPEND _toolchain_contents "set(CMAKE_CXX_COMPILER_LAUNCHER \"@CMAKE_CXX_COMPILER_LAUNCHER@\")\n")
  string(APPEND _toolchain_contents "set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT \"@CMAKE_MSVC_DEBUG_INFORMATION_FORMAT@\")\n")
  if(MSVC AND compiler_toolchain)
    # The system compiler and the toolchain compiler are incompatible, so we
    # define flags from scratch for the toolchain compiler.
    #
    # Each ROCm project typically has its own `toolchain-windows.cmake` file
    # that we are bypassing here. Some projects additionally set platform or
    # configuration-specific options in `rmake.py`. If any flags are load
    # bearing we can either add them to all projects or source those flags from
    # the projects themselves more locally.
    string(APPEND _toolchain_contents "set(CMAKE_C_FLAGS_INIT )\n")
    string(APPEND _toolchain_contents "set(CMAKE_CXX_FLAGS_INIT \"-DWIN32 -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS -DNOMINMAX -fms-extensions -fms-compatibility -D_ENABLE_EXTENDED_ALIGNED_STORAGE \")\n")
    string(APPEND _toolchain_contents "set(CMAKE_EXE_LINKER_FLAGS_INIT )\n")
    string(APPEND _toolchain_contents "set(CMAKE_SHARED_LINKER_FLAGS_INIT )\n")
  else()
    # The system compiler and the toolchain compiler are compatible, so we can
    # simply forward flags from the system compiler to the toolchain compiler.
    string(APPEND _toolchain_contents "set(CMAKE_C_FLAGS_INIT \"@CMAKE_C_FLAGS@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_CXX_FLAGS_INIT \"@CMAKE_CXX_FLAGS@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_EXE_LINKER_FLAGS_INIT \"@CMAKE_EXE_LINKER_FLAGS@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_SHARED_LINKER_FLAGS_INIT \"@CMAKE_SHARED_LINKER_FLAGS@\")\n")
  endif()

  # Customize debug info generation.
  if(THEROCK_MINIMAL_DEBUG_INFO)
    # System toolchain can be either MSVC or another system compiler.
    if(MSVC AND NOT compiler_toolchain)
      # Set MSVC style debug options.
      # TODO: For now, just set the default.
    elseif((NOT compiler_toolchain AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
           OR compiler_toolchain)
      # If the system compiler and GCC/clang or an explicit toolchain (which are all
      # LLVM based).
      string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_DEBUG \" -g1 -gz\")\n")
      string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO \" -g1 -gz\")\n")
      string(APPEND _toolchain_contents "string(APPEND CMAKE_C_FLAGS_DEBUG \" -g1 -gz\")\n")
      string(APPEND _toolchain_contents "string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO \" -g1 -gz\")\n")
    else()
      message(WARNING "Cannot setup THEROCK_MINIMAL_DEBUG_INFO mode for unknown compiler")
    endif()
  endif()

  if(NOT compiler_toolchain)
    # Make any additional customizations if no toolchain specified.
  elseif(compiler_toolchain STREQUAL "amd-llvm" OR compiler_toolchain STREQUAL "amd-hip")
    # The "amd-llvm" and "amd-hip" toolchains are configured very similarly so
    # we commingle them, but they are different:
    #   "amd-llvm": Just the base LLVM compiler and device libraries. This
    #     doesn't know anything about hip (i.e. it doesn't have hipconfig, etc).
    #   "amd-hip": Superset of "amd-llvm" which also includes hipcc, hip headers,
    #     and hip version info. This has hipconfig in it.
    # The main difference is that for "amd-llvm", we derive the configuration from
    # the amd-llvm project's dist/ tree. And for "amd-hip", from the hip-clr
    # project (which has runtime dependencies on the underlying toolchain).
    if(compiler_toolchain STREQUAL "amd-hip")
      set(_toolchain_subproject "hip-clr")
    else()
      set(_toolchain_subproject "amd-llvm")
    endif()
    _therock_assert_is_cmake_subproject("${_toolchain_subproject}")
    get_target_property(_amd_llvm_dist_dir "${_toolchain_subproject}" THEROCK_DIST_DIR)
    get_target_property(_amd_llvm_stamp_dir "${_toolchain_subproject}" THEROCK_STAMP_DIR)
    # Add a dependency on the toolchain's dist
    set(AMD_LLVM_C_COMPILER "${_amd_llvm_dist_dir}/lib/llvm/bin/clang${CMAKE_EXECUTABLE_SUFFIX}")
    set(AMD_LLVM_CXX_COMPILER "${_amd_llvm_dist_dir}/lib/llvm/bin/clang++${CMAKE_EXECUTABLE_SUFFIX}")
    set(AMD_LLVM_LINKER "${_amd_llvm_dist_dir}/lib/llvm/bin/lld${CMAKE_EXECUTABLE_SUFFIX}")
    set(_amd_llvm_cxx_flags_spaces )
    string(JOIN " " _amd_llvm_cxx_flags_spaces ${THEROCK_AMD_LLVM_DEFAULT_CXX_FLAGS})

    list(APPEND _compiler_toolchain_addl_depends "${_amd_llvm_stamp_dir}/dist.stamp")
    # We inject a toolchain root into the subproject so that magic overrides can
    # use it (i.e. for old projects that require path munging, etc).
    string(APPEND _toolchain_contents "set(THEROCK_TOOLCHAIN_ROOT \"${_amd_llvm_dist_dir}\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_C_COMPILER \"@AMD_LLVM_C_COMPILER@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_CXX_COMPILER \"@AMD_LLVM_CXX_COMPILER@\")\n")
    string(APPEND _toolchain_contents "set(CMAKE_LINKER \"@AMD_LLVM_LINKER@\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" ${_amd_llvm_cxx_flags_spaces}\")\n")

    if(THEROCK_VERBOSE)
      string(JOIN " " _filtered_gpu_targets_spaces ${_filtered_gpu_targets})
      message(STATUS "Compiler toolchain ${compiler_toolchain}:")
      string(APPEND CMAKE_MESSAGE_INDENT "  ")
      message(STATUS "CMAKE_C_COMPILER = ${AMD_LLVM_C_COMPILER}")
      message(STATUS "CMAKE_CXX_COMPILER = ${AMD_LLVM_CXX_COMPILER}")
      message(STATUS "CMAKE_LINKER = ${AMD_LLVM_LINKER}")
      message(STATUS "GPU_TARGETS = ${_filtered_gpu_targets_spaces}")
    endif()
  else()
    message(FATAL_ERROR "Unsupported COMPILER_TOOLCHAIN = ${compiler_toolchain} (supported: 'amd-llvm' or none)")
  endif()

  # Configure additional HIP dependencies.
  if (compiler_toolchain STREQUAL "amd-hip")
    _therock_assert_is_cmake_subproject("hip-clr")
    get_target_property(_hip_dist_dir hip-clr THEROCK_DIST_DIR)
    get_target_property(_hip_stamp_dir hip-clr THEROCK_STAMP_DIR)
    # Add a dependency on HIP's stamp.
    set(_amd_llvm_device_lib_path "${_amd_llvm_dist_dir}/lib/llvm/amdgcn/bitcode")
    list(APPEND _compiler_toolchain_addl_depends "${_hip_stamp_dir}/dist.stamp")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --hip-path=@_hip_dist_dir@\")\n")
    string(APPEND _toolchain_contents "string(APPEND CMAKE_CXX_FLAGS_INIT \" --hip-device-lib-path=@_amd_llvm_device_lib_path@\")\n")
    if(THEROCK_VERBOSE)
      message(STATUS "HIP_DIR = ${_hip_dist_dir}")
    endif()
  endif()

  set(_compiler_toolchain_addl_depends "${_compiler_toolchain_addl_depends}" PARENT_SCOPE)
  set(_compiler_toolchain_init_contents "${_compiler_toolchain_init_contents}" PARENT_SCOPE)
  set(_build_env_pairs "${_build_env_pairs}" PARENT_SCOPE)
  file(CONFIGURE OUTPUT "${toolchain_file}" CONTENT "${_toolchain_contents}" @ONLY ESCAPE_QUOTES)
endfunction()
