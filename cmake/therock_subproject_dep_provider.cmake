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
#   THEROCK_STRICT_PROVIDED_PACKAGES: All packages that the super-project knows
#     about. If a sub-project attempts to resolve one of these without a
#     proper declared dependency, it will error.
# See: _therock_cmake_subproject_setup_deps which assembles these variables

block(PROPAGATE CMAKE_PREFIX_PATH)
  set(prefix_extensions)
  if(THEROCK_PKG_CONFIG_DIRS)
    foreach(_dir ${THEROCK_PKG_CONFIG_DIRS})
      # There is a mismatch between THEROCK_PKG_CONFIG_DIRS, which specifies
      # a full path to the .pc file (i.e. /some/path/lib/pkgconfig) versus
      # how CMAKE_PREFIX_PATH is parsed for the same. The latter expects each
      # path to be to the prefix and it will probe sub-directory 'lib/pkgconfig'
      # (actually from an allow list of suitable prefix extensions) to determine
      # if the path will be extended and propagated to pkg-config. Therefore,
      # we need to back up to the actual prefix path. For super-project
      # pkg-config dirs (which is what we are dealing with here), this will
      # always be two levels up (i.e. lib/pkgdata). We could add more validation
      # of this, but it is actually somewhat open ended to determine *which*
      # two levels it is (i.e. lib32 vs lib, share vs lib, etc). But is is
      # always two.
      cmake_path(GET _dir PARENT_PATH _dir)
      cmake_path(GET _dir PARENT_PATH _dir)
      list(APPEND prefix_extensions "${_dir}")
    endforeach()
    message(STATUS "Sub-project pkg-config prefixes: ${prefix_extensions}")
    list(APPEND CMAKE_PREFIX_PATH ${prefix_extensions})
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
    if("${package_name}" IN_LIST THEROCK_STRICT_PROVIDED_PACKAGES)
      cmake_policy(POP)
      message(FATAL_ERROR "Project contains find_package(${package_name}) for a package availabe in the super-project but not declared: Add a BUILD_DEPS or RUNTIME_DEPS appropriately")
    endif()
    cmake_policy(POP)
    message(STATUS "Resolving system find_package(${package_name}) (not found in super-project ${THEROCK_PROVIDED_PACKAGES})")
    set(_therock_has_preserved_module_path FALSE)
    find_package(${package_name} ${ARGN} BYPASS_PROVIDER)
    if(_therock_has_preserved_module_path)
      set(CMAKE_MODULE_PATH "${_therock_preserved_module_path}")
    endif()
  endif()
endmacro()

if(THEROCK_USE_SAFE_DEPENDENCY_PROVIDER AND THEROCK_PROVIDED_PACKAGES)
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
    "BYPASS_PROVIDER;CONFIG;MODULE;NO_DEFAULT_PATH;NO_CMAKE_PATH;NO_CMAKE_ENVIRONMENT_PATH;NO_SYSTEM_ENVIRONMENT_PATH;NO_CMAKE_PACKAGE_REGISTRY"
    ""
    "HINTS;PATHS"
  )
  if(NOT superproject_path)
    message(FATAL_ERROR "Super-project package path not found for ${package_name}")
  endif()

  # The signature of MODULE vs DEFAULT mode is different, so switch.
  set(_rewritten ${UNUSED_UNPARSED_ARGUMENTS})
  # Some very old code uses explicit MODULE mode, which forces the basic
  # signature (*cough* old FindHIP based junk). In this mode, there is no way
  # to explicitly indicate a search path in the signature (and other options
  # are illegal), so we fork on this and hard code the CMAKE_MODULE_PATH
  # in the parent scope, also stashing the original and setting an indicator
  # to restore it. This is not *the best* thing to be doing, but since this
  # is a controlled ecosystem and used for very old things, we tolerate the
  # pitfalls.
  if(UNUSED_MODULE)
    # Explicit find module.
    list(APPEND _rewritten MODULE BYPASS_PROVIDER)
    set(_therock_has_preserved_module_path TRUE PARENT_SCOPE)
    set(_therock_preserved_module_path "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
    set(CMAKE_MODULE_PATH "${superproject_path}" PARENT_SCOPE)
  else()
    # By default, assume a FULL signature.
    if(UNUSED_CONFIG)
      list(APPEND _rewritten "CONFIG")
    endif()
    list(APPEND _rewritten BYPASS_PROVIDER NO_DEFAULT_PATH PATHS ${superproject_path})
  endif()

  list(JOIN _rewritten " " _rewritten_pretty)
  message(STATUS "Resolving super-project find_package(${_rewritten_pretty})")
  set(_therock_rewritten_superproject_find_package_sig ${_rewritten} PARENT_SCOPE)
endfunction()

# Action at a distance: sanitizers need to be able to find their runtime libraries,
# so if configuring a sanitizer, we set the THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH
# variable during toolchain setup. Now in the sub-project, we have access to clang
# and can ask it what its resource dir is. We use this to form a proper relative
# path and append it to the default RPATH dirs.
if(THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH)
  block(PROPAGATE THEROCK_PRIVATE_INSTALL_RPATH_DIRS THEROCK_PRIVATE_BUILD_RPATH_DIRS)
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
      OUTPUT_VARIABLE _abs_resource_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
    # This will yield an absolute path like: /some/long/path/lib/llvm/lib/clang/20
    # The version number on the end is all we want as we will form the correct
    # prefix relative path from that.
    # The actual libraries live under lib/linux in that directory. We shamelessly
    # just hardcode "linux" since that is the only system we do RPATH munging for.
    cmake_path(GET _abs_resource_dir FILENAME _clang_version)
    set(_prefix_resource_dir "lib/llvm/lib/clang/${_clang_version}")
    list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "${_prefix_resource_dir}/lib/linux")
    # Build tree needs absolute paths to the resource dir.
    list(APPEND THEROCK_PRIVATE_BUILD_RPATH_DIRS "${_abs_resource_dir}/lib/linux")
    message(STATUS "Added clang resource dir to RPATH: ${_prefix_resource_dir} (since sanitizer enabled)")
  endblock()
endif()

# Setup THEROCK_SANITIZER_LAUNCHER.
block(PROPAGATE THEROCK_SANITIZER_LAUNCHER)
  set(THEROCK_SANITIZER_LAUNCHER)
  if(LINUX AND THEROCK_SANITIZER STREQUAL "ASAN")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
      set(_arch_suffix "x86_64")
    else()
      message(FATAL_ERROR "Unknown processor ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} "--print-file-name=libclang_rt.asan-${_arch_suffix}.so"
      OUTPUT_VARIABLE _rt_path
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
    set(THEROCK_SANITIZER_LAUNCHER
      "${CMAKE_COMMAND}" -E env "LD_PRELOAD=${_rt_path}" --
    )
    message(STATUS "Sanitizer launcher: ${THEROCK_SANITIZER_LAUNCHER}")
  endif()
endblock()
