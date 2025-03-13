# find_package dependency provider
# This is injected into sub-projects that contain dependencies. It runs in a
# context with the following variables defined at the top level:
#   THEROCK_PROVIDED_PACKAGES: Package names that are to be provided from the
#     super-project
#   THEROCK_PACKAGE_DIR_${package_name}: Directory in the super-project to
#     resolve the dependency.
#   THEROCK_IGNORE_PACKAGES: Packages to ignore, even if they are in
#     THEROCK_PROVIDED_PACKAGES, falling back to the system resolver.
#   THEROCK_PKG_CONFIG_DIRS: Directories to search for .pc files.
# See: _therock_cmake_subproject_setup_deps which assembles these variables

block()
  if(THEROCK_PKG_CONFIG_DIRS)
    if(WIN32)
      set(_sep ";")
    else()
      set(_sep ":")
    endif()
    set(_accum)
    foreach(_dir ${THEROCK_PKG_CONFIG_DIRS})
      if(_accum)
        string(APPEND _accum "${_sep}")
      endif()
      string(APPEND _accum "${_dir}")
    endforeach()
    set(ENV{PKG_CONFIG_PATH} "${_accum}${_sep}$ENV{PKG_CONFIG_PATH}")
    message(STATUS "Sub-project PKG_CONFIG_PATH: $ENV{PKG_CONFIG_PATH}")
  endif()
endblock()

macro(therock_dependency_provider method package_name)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0057 NEW)
  if("${package_name}" IN_LIST THEROCK_PROVIDED_PACKAGES AND NOT
     "${package_name}" IN_LIST THEROCK_IGNORE_PACKAGES)
    cmake_policy(POP)
    # It is quite hard to completely neuter find_package so that for an
    # arbitrary signature it will only attempt to find from one specified path.
    # This is important because it "latches" and if any find_package manages
    # to escape to the system, it will likely find a library from outside the
    # super-project, which can cause all kinds of hard to diagnose issues.
    # For background, read carefully:
    # https://cmake.org/cmake/help/latest/command/find_package.html#config-mode-search-procedure
    # We opt-to rewrite the signature, removing any options that connote an
    # implicit search behavior and then rewrite the signature to be explicit.
    # We further do this in a function to avoid macro namespace pollution, since
    # the find_package itself must be evaluated in the caller-scope.
    therock_reparse_super_project_find_package(
      "${THEROCK_PACKAGE_DIR_${package_name}}" "${package_name}" ${ARGN})
    find_package(${_therock_rewritten_superproject_find_package_sig})
  else()
    cmake_policy(POP)
    message(STATUS "Resolving system find_package(${package_name}) (not found in super-project ${THEROCK_PROVIDED_PACKAGES})")
    find_package(${package_name} ${ARGN} BYPASS_PROVIDER)
  endif()
endmacro()
if(THEROCK_PROVIDED_PACKAGES)
  message(STATUS "Resolving packages from super-project: ${THEROCK_PROVIDED_PACKAGES}")
  cmake_language(
    SET_DEPENDENCY_PROVIDER therock_dependency_provider
    SUPPORTED_METHODS FIND_PACKAGE
  )
endif()

function(therock_reparse_super_project_find_package superproject_path package_name)
  # We parse the arguments we want dropped from the find_package and then use
  # what was unparsed.
  cmake_parse_arguments(PARSE_ARGV 1 UNUSED
    "BYPASS_PROVIDER;CONFIG;NO_DEFAULT_PATH;NO_CMAKE_PATH;NO_CMAKE_ENVIRONMENT_PATH;NO_SYSTEM_ENVIRONMENT_PATH;NO_CMAKE_PACKAGE_REGISTRY"
    ""
    "HINTS;PATHS"
  )
  if(NOT superproject_path)
    message(FATAL_ERROR "Super-project package path not found for ${package_name}")
  endif()

  set(_rewritten ${UNUSED_UNPARSED_ARGUMENTS})
  list(APPEND _rewritten BYPASS_PROVIDER NO_DEFAULT_PATH PATHS ${superproject_path})
  list(JOIN _rewritten " " _rewritten_pretty)
  message(STATUS "Resolving super-project find_package(${_rewritten_pretty})")
  set(_therock_rewritten_superproject_find_package_sig ${_rewritten} PARENT_SCOPE)
endfunction()
