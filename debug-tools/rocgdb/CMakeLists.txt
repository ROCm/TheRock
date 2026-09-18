# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This is a CMake wrapper to build rocgdb from within TheRock. Since GDB's
# build system is autotools-based, we need the wrapper to call the right
# functions and provide the proper include and library paths.
#
# We also take care of cycling through the available Python versions and
# building gdb against each of those.

# Available defines to be used when building rocgdb:
#
# BUILD_TESTING
#
# Enable integrity tests and installation of ROCgdb's testsuite files.
# Usually ON by default in cmake, unless set otherwise.
#
# If set to ON, adds a number of ROCgdb build integrity tests that will
# validate the expected files exist. It also triggers the installation of
# the ROCgdb testsuite files. If set to OFF, no build integrity tests will
# be included and no testsuite files will be installed.
#
#
# PATCHELF
#
# Contains the path to the patchelf utility. This must be passed so
# post-install RPATH patching can take place.
#
#
# ROCGDB_SOURCE_DIR
#
# Specifies the location of the ROCgdb sources. Used by TheRock to build
# ROCgdb from an alternative source location.
#
#
# ROCM_VERSION
#
# Specifies the ROCM version we are building for. If empty or non-existent
# we default to "ROCm development".
#
#
# SHARED_PYTHON_EXECUTABLES
#
# A semicolon-separated list of paths to Python executables that support
# linking against a shared libpython. If empty or non-existent only a
# Pythonless rocgdb (rocgdb-pynone) will be built. Otherwise one rocgdb
# will be built for each of the Python entries on the list, as long as
# each entry supports linking against libpython.  The super-project passes
# it empty on Windows, where only the Pythonless rocgdb is built.
#
# Example: SHARED_PYTHON_EXECUTABLES="/opt/python-shared/cp310-cp310/bin/python3"


cmake_minimum_required(VERSION 3.25)
project(rocgdb C CXX)

include(ExternalProject)
include(ProcessorCount)

# therock_autotools_shell(), which works out how to run configure and make for
# the host being built on.  Pulls in the MSYS2 helper on Windows.
include("${THEROCK_SOURCE_DIR}/cmake/therock_autotools.cmake")

# Function to find a library package and extract its install prefix.
# Uses find_package() which respects CMAKE_PREFIX_PATH and TheRock's
# dependency provider system.
#
# Usage:
#   find_library_prefix(<package_name> <target_name> <output_var>)
#
# Example:
#   find_library_prefix(gmp gmp::gmp GMP_PREFIX)
#
function(find_library_prefix package_name target_name output_var)
  find_package(${package_name} REQUIRED)

  if(TARGET ${target_name})
    # Extract the library location from the imported target
    get_target_property(_lib_location ${target_name} IMPORTED_LOCATION)
    if(_lib_location)
      # Get the directory containing the library (e.g., .../lib)
      cmake_path(GET _lib_location PARENT_PATH _lib_dir)
      # Get the install prefix (e.g., .../lib -> ...)
      cmake_path(GET _lib_dir PARENT_PATH _prefix)
      set(${output_var} "${_prefix}" PARENT_SCOPE)
      message(STATUS "Found ${package_name} at prefix: ${_prefix}")
    else()
      message(FATAL_ERROR "Found ${package_name} target but could not extract location")
    endif()
  else()
    message(FATAL_ERROR "Required package ${package_name} not found.")
  endif()
endfunction()

# Download the ROCgdb CI helper scripts from a single amd-staging commit into
# dest_dir. Resolves the branch first so a push landing between the two
# downloads cannot give a mismatched pair.
#
# Use curl rather than file(DOWNLOAD): file(DOWNLOAD)'s STATUS is the libcurl
# transfer code, so HTTP 404/500 responses are reported as success and the
# error-page body is written to disk. curl --fail returns non-zero on
# HTTP >= 400, and verifies TLS by default.
function(_rocgdb_download_ci_scripts dest_dir)
  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" ls-remote
      https://github.com/ROCm/ROCgdb.git refs/heads/amd-staging
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE git_output
    ERROR_VARIABLE  git_error
  )
  string(REGEX MATCH "^[0-9a-f]+" ci_scripts_sha "${git_output}")
  if(NOT git_result EQUAL 0 OR NOT ci_scripts_sha)
    message(FATAL_ERROR "Failed to resolve ROCm/ROCgdb amd-staging: ${git_error}")
  endif()
  message(STATUS "Downloading CI scripts from ROCm/ROCgdb at ${ci_scripts_sha}")

  find_program(CURL_EXECUTABLE curl REQUIRED)
  foreach(script ${ARGN})
    set(script_path "${dest_dir}/${script}")
    set(script_url "https://raw.githubusercontent.com/ROCm/ROCgdb/${ci_scripts_sha}/.github/scripts/${script}")
    set(download_success FALSE)
    foreach(attempt RANGE 1 3)
      message(STATUS "Attempting to download ${script} (attempt ${attempt}/3)")
      execute_process(
        COMMAND "${CURL_EXECUTABLE}"
          --fail --silent --show-error --location
          --max-time 60
          --output "${script_path}"
          "${script_url}"
        RESULT_VARIABLE curl_result
        ERROR_VARIABLE  curl_error
      )
      if(curl_result EQUAL 0)
        set(download_success TRUE)
        message(STATUS "Successfully downloaded ${script} to ${script_path}")
        break()
      else()
        message(WARNING "Download attempt ${attempt} failed: ${curl_error}")
        file(REMOVE "${script_path}")
      endif()
    endforeach()

    if(NOT download_success)
      message(FATAL_ERROR "Failed to download ${script} after 3 attempts")
    endif()
  endforeach()
endfunction()

# By default install the ROCgdb testsuite files.
option(ENABLE_TESTS "Enable installation of ROCgdb testsuite files" OFF)

# Upstream build mode: skips AMD-specific files not yet in upstream master
# (NOTICES.txt, roccoremerge program/manpage).
option(ROCGDB_UPSTREAM_BUILD "Building from upstream master (skips AMD-specific files)" OFF)

# When the rocgdb source tree does not carry .github/scripts/test_rocgdb.py,
# rocjitsu_emulator.py or rocgdb_emulate.py (e.g. an upstream branch), fetch
# them from ROCm/ROCgdb amd-staging at configure time so that they can be
# installed for TheRock CI to invoke.
option(ROCGDB_DOWNLOAD_CI_SCRIPT "Download test_rocgdb.py, rocjitsu_emulator.py and rocgdb_emulate.py from amd-staging if missing in source tree" OFF)

# Find standard build tools.  On Windows the tools are invoked through the MSYS2
# login shell that therock_autotools_shell() sets up, so make is named bare and
# left for that shell's PATH to resolve: the host PATH may carry no make at all,
# or one that cannot drive a GNU autotools tree.
if(WIN32)
  set(MAKE_EXECUTABLE make)
else()
  find_program(MAKE_EXECUTABLE NAMES make REQUIRED)
endif()

ProcessorCount(PAR_JOBS)

# Configure build type flags.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Standard flags are handled by CMAKE_C_FLAGS, CMAKE_C_FLAGS_DEBUG, etc.
# Initialize with base flags and build-type-specific flags.
set(CUSTOM_C_FLAGS "${CMAKE_C_FLAGS}")
set(CUSTOM_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Append build-type-specific flags.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type_upper)
if(_build_type_upper)
  string(APPEND CUSTOM_C_FLAGS " ${CMAKE_C_FLAGS_${_build_type_upper}}")
  string(APPEND CUSTOM_CXX_FLAGS " ${CMAKE_CXX_FLAGS_${_build_type_upper}}")
endif()

# Set custom flags for Debug builds.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  string(APPEND CUSTOM_C_FLAGS " -O0 -g3")
  string(APPEND CUSTOM_CXX_FLAGS " -O0 -g3")
endif()

# Harden against stack clash attacks (not supported on all platforms).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  string(APPEND CUSTOM_C_FLAGS " -fstack-clash-protection")
  string(APPEND CUSTOM_CXX_FLAGS " -fstack-clash-protection")
endif()

message(STATUS "Build type set to: ${CMAKE_BUILD_TYPE}")

# amd-dbgapi is always a required dependency for ROCgdb
find_package(amd-dbgapi REQUIRED CONFIG)

# Fetch the install prefixes for the library and executables rocgdb
# depends on. When THEROCK_BUNDLE_SYSDEPS is OFF, these are left empty and
# GDB's configure script discovers the libraries via pkg-config instead.
set(NCURSES_RPATH "")
set(NCURSES_INCLUDE "")
if(THEROCK_BUNDLE_SYSDEPS)
  find_library_prefix(gmp gmp::gmp GMP_PREFIX)
  find_library_prefix(mpfr mpfr::mpfr MPFR_PREFIX)
  # --with-expat takes no prefix: expat resolves through the -L that
  # CMAKE_SHARED_LINKER_FLAGS carries for every bundled sysdep.
  find_package(expat REQUIRED)
  find_library_prefix(ncurses ncurses::ncurses NCURSES_PREFIX)

  # We need to explicitly pass ncurses' include path and set the rpath for
  # the library. The reason we need to explicitly pass rpath is because gdb's
  # configure assumes ncurses is always picked up from the default system
  # location, or that the library is always within reach.
  #
  # Since we're building our own ncurses and it is in a non-standard location,
  # the rpath assures the configure tests properly locate the library.
  # Extract include and lib directories from the ncurses imported target.
  if(TARGET ncurses::ncurses)
    get_target_property(_ncurses_includes ncurses::ncurses INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_ncurses_location ncurses::ncurses IMPORTED_LOCATION)
    if(_ncurses_includes)
      set(NCURSES_INCLUDE "-I${_ncurses_includes}")
    endif()
    if(_ncurses_location)
      cmake_path(GET _ncurses_location PARENT_PATH _ncurses_lib_dir)
      set(NCURSES_RPATH "-Wl,-rpath,${_ncurses_lib_dir}")
    endif()
  endif()

  # Adjust the compilation flags with the bundled ncurses include path.
  if(NCURSES_INCLUDE)
    set(CUSTOM_C_FLAGS "${CUSTOM_C_FLAGS} ${NCURSES_INCLUDE}")
    set(CUSTOM_CXX_FLAGS "${CUSTOM_CXX_FLAGS} ${NCURSES_INCLUDE}")
  endif()
endif()

# ###########################################################################
# Windows specifics.
#
# rocgdb.exe must not rely on DLLs from MSYS2, which is a build-time dependency
# here, not something a user of the ROCm distribution has.  Every dependency is
# therefore either linked statically or resolved to one of our own DLLs, and
# each of the Windows-only settings below exists to hold that line:
# --with-static-standard-libraries for libstdc++/libgcc, a static winpthread,
# static .a for the sysdeps built under MSYS2, and our private
# rocm_sysdeps_*.dll for the rest.  The only DLLs rocgdb.exe ends up importing
# are the amd-dbgapi one, whose name carries its major and minor version
# (rocm-dbgapi-0-84.dll and so on), rocm_sysdeps_*.dll and OS DLLs.  Check it
# after changing anything here with "LC_ALL=C objdump -p rocgdb.exe | grep 'DLL
# Name'".
# ###########################################################################
if(WIN32)
  # These two are looked up here instead of in the shared block above, because
  # that block runs on every host and these do not apply to all of them.
  #
  # iconv lives in glibc on Linux, so the libiconv sysdep is built for Windows
  # only and find_package(iconv) has nothing to resolve elsewhere.
  #
  # liblzma is built on both hosts, but only here does its include directory
  # have to be named explicitly.  On Linux the probe in gdb's configure locates
  # liblzma.so through the -L that CMAKE_SHARED_LINKER_FLAGS already carries,
  # and derives the matching include directory from it -- which is exactly what
  # it cannot do on Windows, for the reason spelled out below.
  #
  # The liblzma package is xz's own CMake config, whose liblzma::liblzma is the
  # DLL that rocgdb links.  Not LibLZMA: TheRock's config of that name describes
  # the static library, which is for MSVC consumers only.
  find_package(liblzma REQUIRED)
  find_package(iconv REQUIRED)

  # gdb's configure finds expat, lzma and iconv by a link probe
  # (-lexpat/-llzma/-liconv) plus an -I for each header directory.  It cannot
  # consume a CMake imported target, so read each one's include directory from
  # the target find_package provides (the sysdep configs set
  # INTERFACE_INCLUDE_DIRECTORIES) and pass it as -I.  ncurses needs no entry
  # here, being covered by NCURSES_INCLUDE above.  The library directories need
  # no equivalent gathering: the superproject already appends a -L for every
  # runtime dep to CMAKE_SHARED_LINKER_FLAGS, built from each dep's
  # INTERFACE_LINK_DIRS.  gmp and mpfr are handled by --with-gmp=/--with-mpfr=,
  # which emit their own -I/-L.
  #
  # Do not drop lzma from this list on the grounds that the gettext lib-link.m4
  # macro derives a header directory by itself.  It does that only for a library
  # its probe actually locates, searching each -L directory for the file names
  # lib<name>.a and lib<name>.dll.  That finds libexpat.a and libiconv.a, so
  # expat and iconv would indeed be covered, but liblzma ships here as the
  # import library lzma.lib, which the probe does not match -- ld itself accepts
  # that name on PE targets, so the link still succeeds and only the header
  # search comes up empty.
  set(_sysdep_includes "")
  foreach(_tgt expat::expat liblzma::liblzma iconv::iconv)
    get_target_property(_inc "${_tgt}" INTERFACE_INCLUDE_DIRECTORIES)
    if(_inc)
      string(APPEND _sysdep_includes " -I${_inc}")
    endif()
  endforeach()

  # _WIN32_WINNT must be at least 0x600 (Vista) for "#include <mutex>" to
  # work.  We only support Windows 10 and up, though, so define it as that,
  # which also makes some gnulib code a bit more efficient.
  # -DNCURSES_STATIC makes ncurses_dll.h declare plain symbols instead of
  # __declspec(dllimport): our ncurses is a static .a, and the default dllimport
  # declarations would leave __imp_* thunks unresolved at link time.
  #
  # These go in CFLAGS/CXXFLAGS, not CPPFLAGS: gdb's combined-tree top-level
  # Makefile hands CFLAGS to the configures it runs in each subdirectory, but
  # not the host CPPFLAGS, so a -D or -I put in CPPFLAGS would never reach
  # gdb/.
  set(_win_flags "-D_WIN32_WINNT=0xA00 -DNCURSES_STATIC${_sysdep_includes}")
  string(APPEND CUSTOM_C_FLAGS " ${_win_flags}")
  string(APPEND CUSTOM_CXX_FLAGS " ${_win_flags}")
endif()

# Add RPATH entries for sanitizer runtime libraries if needed.
# THEROCK_PRIVATE_BUILD_RPATH_DIRS contains absolute paths to directories
# needed during build/configure (e.g., clang sanitizer runtime).
set(SANITIZER_RPATH "")
if(DEFINED THEROCK_PRIVATE_BUILD_RPATH_DIRS AND NOT "${THEROCK_PRIVATE_BUILD_RPATH_DIRS}" STREQUAL "")
  foreach(build_rpath_dir ${THEROCK_PRIVATE_BUILD_RPATH_DIRS})
    string(APPEND SANITIZER_RPATH "-Wl,-rpath,${build_rpath_dir} ")
  endforeach()
  string(STRIP "${SANITIZER_RPATH}" SANITIZER_RPATH)
  message(STATUS "Adding sanitizer RPATH for configure/build: ${SANITIZER_RPATH}")
endif()

# Adjust LDFLAGS.
if(WIN32)
  # Statically link winpthread, but only when the toolchain's C++ threading
  # model is posix.  gdb's C++ pulls in libstdc++'s threading, and on a
  # posix-threads ucrt64 (the current default) that references winpthread.
  # --with-static-standard-libraries makes libstdc++/libgcc static but leaves
  # winpthread resolving to the ucrt64 libwinpthread-1.dll's import library.
  #
  # gdb links through libtool, which defeats the obvious flag approaches: a
  # -Wl,--whole-archive -lwinpthread -Wl,--no-whole-archive group does not
  # survive it.  libtool hoists the -lwinpthread out of the bracket into its own
  # deplib list, emitted after the objects in dynamic mode, leaving
  # --whole-archive bracketing nothing.  What libtool does forward untouched is
  # library search (-L/-l), so use the trick the static sysdeps already rely on:
  # a -L directory offering only the static archive, searched ahead of
  # /ucrt64/lib, which also ships the .dll.a.  ld takes the first search
  # directory with any match for a name, so a directory holding just the .a wins
  # and the .dll.a is never reached.
  #
  # Two names are needed.  A posix-model gcc adds -lpthread to every link (its
  # link spec has %{!no-pthread:-lpthread}, so gdb's -pthread changes
  # nothing), while gdb's deplibs carry -lwinpthread.  In mingw-w64 both are
  # the same static library under two names, so shadow both, whichever ld
  # searches first.
  #
  # The shadow files are tiny GNU ld linker scripts naming the real archive by
  # absolute path, not copies.  ld type-sniffs a text file named lib*.a as a
  # script, so this adds no stale binary to the build tree and tracks whatever
  # libwinpthread.a the toolchain currently ships.
  #
  # With the win32 thread model libstdc++ uses native Windows threads and never
  # references winpthread, and gdb itself uses no pthread function on Windows,
  # so this is skipped there.  gdb's -pthread then does add -lpthread, but with
  # nothing referencing it ld pulls in no import, and no DLL.
  execute_process(
    COMMAND "${CMAKE_C_COMPILER}" -v
    ERROR_VARIABLE _gcc_v OUTPUT_QUIET ERROR_STRIP_TRAILING_WHITESPACE)
  # A failed probe would otherwise read as "not posix" and skip the shadowing,
  # leaving rocgdb.exe importing libwinpthread-1.dll with nothing to say so.
  if(NOT _gcc_v MATCHES "Thread model: ")
    message(FATAL_ERROR
      "Could not read the thread model from '${CMAKE_C_COMPILER} -v'")
  endif()
  set(_win_pthread_ldflags "")
  if(_gcc_v MATCHES "Thread model: posix")
    set(_winpthread_static_dir "${CMAKE_CURRENT_BINARY_DIR}/winpthread-static")
    foreach(_wp libpthread libwinpthread)
      # Ask the compiler which archive it would link, instead of assuming where
      # its toolchain keeps them.  gcc prints the bare name back when it finds
      # no such file.
      execute_process(
        COMMAND "${CMAKE_C_COMPILER}" "-print-file-name=${_wp}.a"
        OUTPUT_VARIABLE _wp_archive OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_wp_archive STREQUAL "${_wp}.a")
        message(FATAL_ERROR
          "${CMAKE_C_COMPILER} has no ${_wp}.a to link winpthread statically")
      endif()
      file(WRITE "${_winpthread_static_dir}/${_wp}.a"
        "GROUP ( \"${_wp_archive}\" )\n")
    endforeach()
    set(_win_pthread_ldflags " -L${_winpthread_static_dir}")
  endif()

  # PE has no RPATH, so NCURSES_RPATH, SANITIZER_RPATH and --enable-new-dtags
  # have no counterpart here -- a DLL is found next to the executable that
  # imports it.  CMAKE_SHARED_LINKER_FLAGS already carries a -L for every
  # runtime dep, which is what lets gdb's configure resolve -lexpat and friends
  # against our libraries instead of the msys2 ucrt64 .dll.a copies in
  # /ucrt64/lib (that directory is on gcc's built-in search path, searched after
  # any -L).  Our directories hold the static .a for the autotools sysdeps, and
  # the private DLLs' import libraries for zlib, zstd and liblzma.
  set(LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}${_win_pthread_ldflags}")
else()
  # NCURSES_RPATH and SANITIZER_RPATH are empty when not using bundled sysdeps
  # or sanitizers.
  set(LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NCURSES_RPATH} ${SANITIZER_RPATH} -Wl,--enable-new-dtags")
endif()

# Set conditional configure flags for bundled sysdep prefixes. When using system
# libraries, these are left empty so autotools discovers them via pkg-config.
set(ROCGDB_GMP_FLAG "")
set(ROCGDB_MPFR_FLAG "")
if(THEROCK_BUNDLE_SYSDEPS)
  set(ROCGDB_GMP_FLAG "--with-gmp=${GMP_PREFIX}")
  set(ROCGDB_MPFR_FLAG "--with-mpfr=${MPFR_PREFIX}")
endif()

# Build RPATH segments from THEROCK_PRIVATE_INSTALL_RPATH_DIRS if defined.
# These paths are relative to <root>, but need to be relative to <root>/bin.
# If not defined, fall back to hardcoded sysdeps path for backward compatibility.
set(ROCGDB_EXTRA_RPATH_SEGMENTS "")
if(DEFINED THEROCK_PRIVATE_INSTALL_RPATH_DIRS AND NOT "${THEROCK_PRIVATE_INSTALL_RPATH_DIRS}" STREQUAL "")
  foreach(rpath_dir ${THEROCK_PRIVATE_INSTALL_RPATH_DIRS})
    # Convert path from <root>-relative to <root>/bin-relative by prepending ../
    string(APPEND ROCGDB_EXTRA_RPATH_SEGMENTS ":\\\$ORIGIN/../${rpath_dir}")
  endforeach()
  message(STATUS "RPATH segments from THEROCK_PRIVATE_INSTALL_RPATH_DIRS: ${ROCGDB_EXTRA_RPATH_SEGMENTS}")
elseif(THEROCK_BUNDLE_SYSDEPS)
  # Fallback for older builds without THEROCK_PRIVATE_INSTALL_RPATH_DIRS
  set(ROCGDB_EXTRA_RPATH_SEGMENTS ":\\\$ORIGIN/../lib/rocm_sysdeps/lib")
endif()

# Set some global variables for the configure/build.
set(ROCGDB_BUG_URL "https://github.com/ROCm-Developer-Tools/ROCgdb/issues")

# Configure flags that depend on the host being built for.  They come last on
# the configure line, so they win over anything set earlier.  How the tools are
# *run* is not decided here -- therock_autotools_shell() below works that out
# per variant.
set(ROCGDB_HOST_CONFIGURE_FLAGS "")

if(WIN32)
  # The target is MSVC-ABI because that is what the programs rocgdb debugs are
  # built as.  It differs from the host, which is the mingw triplet the helper
  # supplies: rocgdb itself runs as a native ucrt64 program.
  #
  # The remaining flags trim the combined tree to what Windows ships.
  # --disable-binutils takes coremerge with it (core dumping is not supported on
  # Windows), and --disable-nls and --disable-source-highlight each keep
  # configure from finding a further msys2 library.
  set(ROCGDB_HOST_CONFIGURE_FLAGS
    --target=x86_64-pc-windows-msvc
    --with-static-standard-libraries
    --disable-binutils
    --disable-nls
    --disable-source-highlight)
endif()

# Directories GDB searches for auto-load scripts (and trusts, since the
# safe-path defaults to this same list). Setting --with-auto-load-dir replaces
# GDB's built-in default of "$debugdir:$datadir/auto-load", so that default is
# repeated here. $debugdir and $datadir are expanded by GDB at run time and
# must reach configure unexpanded.
#
# On Linux we also look at the locations where distros install helper scripts
# (e.g. python3-dbg's pythonX.Y-gdb.py and libstdc++'s pretty-printers):
#   /usr/share/gdb/auto-load        Debian, Ubuntu, Fedora, RHEL, SUSE, Arch, Alpine
#   /usr/local/share/gdb/auto-load  Locally installed packages (e.g. from source)
# Fedora, RHEL and SUSE debuginfo packages also place scripts under
# /usr/lib/debug, which is already covered by $debugdir through
# ROCGDB_SEPARATE_DEBUG_INFO_DIR.
set(ROCGDB_AUTO_LOAD_DIR_FLAG "")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ROCGDB_AUTO_LOAD_DIR
      "$debugdir:$datadir/auto-load:/usr/share/gdb/auto-load:/usr/local/share/gdb/auto-load")
  set(ROCGDB_AUTO_LOAD_DIR_FLAG
      "--with-auto-load-dir=${ROCGDB_AUTO_LOAD_DIR}"
      "--with-auto-load-safe-path=${ROCGDB_AUTO_LOAD_DIR}")
endif()

# If the version is not set, set it to a generic value.
if (NOT ROCM_VERSION)
  set(ROCM_BUILD_ID "ROCm development")
else()
  set(ROCM_BUILD_ID "ROCm-${ROCM_VERSION}")
endif()

# List to accumulate all unique build targets.
set(ALL_ROCGDB_TARGETS)

# Whether any variant is built with python.  The python runtime files under the
# data directory come from those variants, so where there are none they are not
# installed and must not be expected.
set(ROCGDB_ANY_PYTHON_VARIANT FALSE)

# Adds a rocgdb build target for a python variant with explicit additional configure flags.
# The special minor_version "none" indicates that the build is without python support.
#
# For variants with minor_version other than "none", libpython_dir must contain
# the path to the shared libpython rocgdb should link against.
macro(add_python_variant minor_version configure_flags libpython_dir)
  if(NOT "${minor_version}" STREQUAL "none")
    set(ROCGDB_ANY_PYTHON_VARIANT TRUE)
  endif()

  # Name of the installed executable for this variant.  Where several python
  # variants can be built, each is given a -py<version> suffix and a wrapper
  # script named plain "rocgdb" picks between them at run time.  Windows builds
  # the no-python variant only and has no use for a POSIX wrapper script, so the
  # executable is named "rocgdb" directly there.
  if(WIN32)
    set(ROCGDB_PROGRAM_NAME "rocgdb")
    set(ROCGDB_PROGRAM_TRANSFORM_FLAG "")
  else()
    set(ROCGDB_PROGRAM_NAME "rocgdb-py${minor_version}")
    set(ROCGDB_PROGRAM_TRANSFORM_FLAG
      "--program-transform-name=s/^rocgdb$/rocgdb-py${minor_version}/")
  endif()

  # Define a unique name for the build directory.
  set(ROCGDB_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocgdb-build-py${minor_version}")
  # Define a unique name for the intermediate install directory.
  set(ROCGDB_INTERMEDIATE_INSTALL_DIR "${ROCGDB_BUILD_DIR}-install")
  set(ROCGDB_TARGET_NAME "rocgdb-py${minor_version}")

  # If we have a valid libpython_dir, adjust the LDFLAGS for this build
  # variant and add an extra RPATH entry for post-install patching.
  #
  # Skip this for system library directories (under /usr, excluding /usr/local).
  # The linker already searches these by default, and explicitly adding them
  # via -L pulls in unrelated system static archives (e.g. libiberty.a from
  # libiberty-dev) that conflict with GDB's bundled copies. GDB's configure
  # finds system libpython on its own via --with-python without needing -L.
  # Non-system libpython (custom builds, /opt, /usr/local, etc.) still needs
  # the explicit flags since those paths are not in the default search order.
  # See https://github.com/ROCm/TheRock/issues/4223
  set(ALL_LDFLAGS ${LDFLAGS})
  set(LIBPYTHON_RPATH "")
  if(NOT "${libpython_dir}" STREQUAL "")
    set(_usr_prefix "/usr")
    set(_usr_local_prefix "/usr/local")
    cmake_path(IS_PREFIX _usr_prefix "${libpython_dir}" NORMALIZE _is_under_usr)
    cmake_path(IS_PREFIX _usr_local_prefix "${libpython_dir}" NORMALIZE _is_under_usr_local)
    if(_is_under_usr AND NOT _is_under_usr_local)
      message(STATUS "-> Skipping extra LDFLAGS for system libpython dir: ${libpython_dir}")
    else()
      set(LIBPYTHON_LDFLAGS "-L${libpython_dir} -Wl,-rpath,${libpython_dir}")
      string(APPEND ALL_LDFLAGS " ${LIBPYTHON_LDFLAGS}")
      set(LIBPYTHON_RPATH ":${libpython_dir}")

      message(STATUS "-> Extra LDFLAGS: ${LIBPYTHON_LDFLAGS}")
      message(STATUS "-> Extra RPATH: ${LIBPYTHON_RPATH}")
    endif()
  endif()

  # These directories are named outright instead of through configure's
  # ${prefix}.  Naming them costs nothing: configure's prefix is this same
  # intermediate install directory, so a "${prefix}/x" expands to exactly what
  # is written here, and gdb relocates a path under its configured prefix
  # relative to the executable at run time either way.
  #
  # A ${prefix} spelling would not survive the trip on Windows anyway.  It has
  # to be escaped to reach configure, and "cmake -E env" rewrites the escaped
  # form to $prefix, which make reads as $(p) followed by "refix" -- the files
  # would install into a stray relative "refix/" directory and the built rocgdb
  # would end up with no data directory at all.
  set(ROCGDB_HTML_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/html")
  set(ROCGDB_PDF_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/doc")
  set(ROCGDB_INFO_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/info/rocgdb")
  set(ROCGDB_DATA_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/rocgdb")

  # The debug-info search path is split on the host's directory separator, which
  # is ';' on Windows -- a ':'-joined list would be read there as one
  # nonsensical path.  There is no system debug-info directory to name on
  # Windows anyway, so leave just our own.
  set(ROCGDB_SEPARATE_DEBUG_INFO_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/lib/debug")
  if(NOT WIN32)
    string(APPEND ROCGDB_SEPARATE_DEBUG_INFO_DIR ":/usr/lib/debug")
  endif()

  # Build AMD-specific install commands (conditionally included).
  set(AMD_SPECIFIC_INSTALL_COMMANDS "")
  if(NOT ROCGDB_UPSTREAM_BUILD)
    list(APPEND AMD_SPECIFIC_INSTALL_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E copy ${ROCGDB_SOURCE_DIR}/gdb/NOTICES.txt ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/doc/
    )
  endif()

  # How to run this variant's build tools, and how to name configure.  Only
  # configure is given the compiler and flag variables -- by the time make runs,
  # configure has baked them into the Makefiles.
  therock_autotools_shell(
    BINARY_DIR "${ROCGDB_BUILD_DIR}"
    SOURCE_DIR "${ROCGDB_SOURCE_DIR}"
    PKG_CONFIG_PATH "${THEROCK_PKG_CONFIG_DIRS}"
    CONFIGURE_ENV
      "CFLAGS=${CUSTOM_C_FLAGS}"
      "CXXFLAGS=${CUSTOM_CXX_FLAGS}"
      "LDFLAGS=${ALL_LDFLAGS}"
    RUN_VAR ROCGDB_RUN
    RUN_CONFIGURE_VAR ROCGDB_RUN_CONFIGURE
    CONFIGURE_VAR ROCGDB_CONFIGURE
    CONFIGURE_ARGS_VAR ROCGDB_HOST_ARGS)

  # binutils is what provides coremerge, and the Windows build does not
  # configure it (see ROCGDB_HOST_CONFIGURE_FLAGS).  gdb installs its gcore
  # script only on hosts where gcore is supported natively, which excludes
  # mingw, so there is nothing to rename there either.
  set(ROCGDB_BINUTILS_INSTALL_COMMANDS "")
  set(ROCGDB_GCORE_RENAME_COMMANDS "")
  if(NOT WIN32)
    list(APPEND ROCGDB_BINUTILS_INSTALL_COMMANDS
      COMMAND ${ROCGDB_RUN} ${MAKE_EXECUTABLE} -s -C binutils install
    )
    list(APPEND ROCGDB_GCORE_RENAME_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E rename
        ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/bin/rocgcore
        ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/bin/rocgcore-py${minor_version}
    )
  endif()

  # ROCGDB_BUILD_DIR is created when this file is configured, by
  # therock_autotools_shell() above, along with what it puts there on Windows
  # (msys2-run.sh, the msys2-home directory, and the therock-srcdir junction
  # for a source on another drive).  Nothing recreates it at build time, so
  # deleting it by hand takes deleting the rocgdb sub-project's
  # stamp/configure.stamp too, or using rocgdb+expunge instead.
  add_custom_target("${ROCGDB_TARGET_NAME}"
    ALL
    VERBATIM
    WORKING_DIRECTORY "${ROCGDB_BUILD_DIR}"
    COMMENT "rocgdb (python ${CURRENT_PYTHON_EXECUTABLE})"

    # Configure
    COMMAND
      ${ROCGDB_RUN_CONFIGURE}
      # The actual configure command.
      ${ROCGDB_CONFIGURE}
        ${ROCGDB_HOST_ARGS}
        "--prefix=${ROCGDB_INTERMEDIATE_INSTALL_DIR}"
        --program-prefix=roc
        ${ROCGDB_PROGRAM_TRANSFORM_FLAG}

        --enable-64-bit-bfd
        --enable-targets=amdgcn-amd-amdhsa
        --enable-tui

        "--htmldir=${ROCGDB_HTML_DIR}"
        "--pdfdir=${ROCGDB_PDF_DIR}"
        "--infodir=${ROCGDB_INFO_DIR}"

        --with-amd-dbgapi
        ${ROCGDB_AUTO_LOAD_DIR_FLAG}
        "--with-bugurl=${ROCGDB_BUG_URL}"
        --with-expat
        "--with-gdb-datadir=${ROCGDB_DATA_DIR}"
        ${ROCGDB_GMP_FLAG}
        --with-lzma
        ${ROCGDB_MPFR_FLAG}
        "--with-pkgversion=${ROCM_BUILD_ID}"
        "--with-separate-debug-dir=${ROCGDB_SEPARATE_DEBUG_INFO_DIR}"
        --with-system-zlib
        --with-zstd

        --disable-gas
        --disable-gdbserver
        --disable-gdbtk
        --disable-gprof
        --disable-gprofng
        --disable-ld
        --disable-shared
        --disable-sim

        --without-babeltrace
        --without-debuginfod
        --without-guile
        --without-intel-pt
        --without-libunwind-ia64
        --without-xxhash
        ${configure_flags}
        ${ROCGDB_HOST_CONFIGURE_FLAGS}

    # Build
    COMMAND
      ${ROCGDB_RUN} ${MAKE_EXECUTABLE} -s -j "${PAR_JOBS}"
    # Install gdb.
    COMMAND
      ${ROCGDB_RUN} ${MAKE_EXECUTABLE} -s -C gdb install
    # Install gdb docs.
    COMMAND
      ${ROCGDB_RUN} ${MAKE_EXECUTABLE} -s -C gdb install-html
    # Install binutils for coremerge and coremerge manpage.
    ${ROCGDB_BINUTILS_INSTALL_COMMANDS}
    # Create doc directory and add AMD-specific files (NOTICES.txt) if not building upstream.
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/doc
    ${AMD_SPECIFIC_INSTALL_COMMANDS}

    # Rename rocgcore to rocgcore-py<python version>.
    ${ROCGDB_GCORE_RENAME_COMMANDS}
  )

  list(APPEND ALL_ROCGDB_TARGETS ${ROCGDB_TARGET_NAME})

  # We only want to install a subset of the files, so we list them here,
  # relative to their intermediate install prefixes.
  # For instance, we don't want to include any binutils files other than
  # roccoremerge.

  # Misc directories.
  set(ROCGDB_MISC_DIRS_TO_INSTALL
    "include"
    "share/doc"
    "share/man/man5"
    "share/rocgdb"
  )

  # Executable files.
  set(ROCGDB_EXEC_FILES_TO_INSTALL
    "bin/${ROCGDB_PROGRAM_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
  )
  if(NOT WIN32)
    list(APPEND ROCGDB_EXEC_FILES_TO_INSTALL "bin/rocgcore-py${minor_version}")
  endif()

  # Misc files.
  set(ROCGDB_MISC_FILES_TO_INSTALL
    "share/info/rocgdb/annotate.info"
    "share/info/rocgdb/gdb.info"
    "share/man/man1/rocgdb.1"
    "share/man/man1/rocgdb-add-index.1"
  )

  # gcore's man page follows the gcore script it documents, so it is absent
  # wherever gcore is not supported natively.
  if(NOT WIN32)
    list(APPEND ROCGDB_MISC_FILES_TO_INSTALL "share/man/man1/rocgcore.1")
  endif()

  # AMD-specific files not yet in upstream master.  coremerge comes from
  # binutils, which the Windows build does not configure.
  if(NOT ROCGDB_UPSTREAM_BUILD AND NOT WIN32)
    list(APPEND ROCGDB_EXEC_FILES_TO_INSTALL "bin/roccoremerge")
    list(APPEND ROCGDB_MISC_FILES_TO_INSTALL "share/man/man1/roccoremerge.1")
  endif()

  # Register the final installation step for this variant's files.
  # This basically moves the variant files from the intermediate install
  # prefix to the final installation location.
  foreach(rocgdb_dir ${ROCGDB_MISC_DIRS_TO_INSTALL})
    # The last "/" is important, as we want to copy the contents from the
    # listed directory into another directory, and not the directory itself.
    install(DIRECTORY ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/${rocgdb_dir}/
            DESTINATION "${rocgdb_dir}"
            USE_SOURCE_PERMISSIONS
            COMPONENT rocgdb)
  endforeach()

  foreach(rocgdb_file ${ROCGDB_EXEC_FILES_TO_INSTALL})
    install(PROGRAMS ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/${rocgdb_file}
            DESTINATION bin
            COMPONENT rocgdb)
  endforeach()

  foreach(rocgdb_file ${ROCGDB_MISC_FILES_TO_INSTALL})
    # Get the directory part of the file.
    get_filename_component(file_dir ${rocgdb_file} DIRECTORY)

    install(FILES ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/${rocgdb_file}
      DESTINATION ${file_dir}
      COMPONENT rocgdb)
  endforeach()

  # Register RPATH patching for this rocgdb variant.  PE has no RPATH: a DLL is
  # found next to the executable that imports it, which is where the
  # distribution already puts them.
  if(NOT WIN32)
    message(STATUS "Registering fixup of ${CMAKE_INSTALL_PREFIX}/bin/${ROCGDB_PROGRAM_NAME}")
    install(CODE "execute_process(
        COMMAND \"${PATCHELF}\" --set-rpath \"\$ORIGIN${ROCGDB_EXTRA_RPATH_SEGMENTS}${LIBPYTHON_RPATH}\"
        \"${CMAKE_INSTALL_PREFIX}/bin/${ROCGDB_PROGRAM_NAME}\"
        COMMAND_ERROR_IS_FATAL ANY
      )"
    COMPONENT rocgdb)
  endif()

  if(BUILD_TESTING)
    # Track for build validation.
    list(APPEND ROCGDB_BUILD_INTEGRITY_FILES
      "bin/${ROCGDB_PROGRAM_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
    if(NOT WIN32)
      list(APPEND ROCGDB_BUILD_INTEGRITY_FILES "bin/rocgcore-py${minor_version}")
    endif()
  endif()
endmacro()

# Determine python versions either from the explicit list of dist python executables
# or from the global Python3_EXECUTABLE.
if(SHARED_PYTHON_EXECUTABLES)
  message(STATUS "Available python-shared: ${SHARED_PYTHON_EXECUTABLES}")
  # Build once for each shared-built python executables we were told to use.
  # Note that the build will fail if libpython is not available in any of these.
  foreach(CURRENT_PYTHON_EXECUTABLE ${SHARED_PYTHON_EXECUTABLES})
    # Query each dist python interpreter for version information.
    execute_process(
      COMMAND "${CURRENT_PYTHON_EXECUTABLE}" -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor};{v.major}.{v.minor}.{v.micro}')"
      OUTPUT_VARIABLE _python_versions
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )
    list(GET _python_versions 0 CURRENT_PYTHON_VERSION)
    list(GET _python_versions 1 CURRENT_PYTHON_FULL_VERSION)
    message(STATUS "Building rocgdb for python ${CURRENT_PYTHON_EXECUTABLE}: Minor=${CURRENT_PYTHON_VERSION}, Full=${CURRENT_PYTHON_FULL_VERSION}")
    message(STATUS "-> Found Python ${CURRENT_PYTHON_FULL_VERSION} at ${CURRENT_PYTHON_EXECUTABLE}")

    # For each Python that supports linking against a shared libpython we
    # fetch the library directory so we can pass a linker flag to rocgdb's
    # configure.
    execute_process(
      COMMAND "${CURRENT_PYTHON_EXECUTABLE}" -c "import sysconfig; libdir=sysconfig.get_config_var('LIBDIR'); print(f'{libdir}')"
      OUTPUT_VARIABLE _libpython_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY
    )

    # Sanity check that we managed to get the libpython path. Otherwise skip
    # this particular build.
    if(_libpython_dir)
      message(STATUS "-> Found libpython path ${_libpython_dir}")

      # Add build instructions for this python-specific rocgdb.
      add_python_variant("${CURRENT_PYTHON_VERSION}" "--with-python=${CURRENT_PYTHON_EXECUTABLE}" "${_libpython_dir}")
    else()
      message(STATUS "Failed to extract libpython directory for ${CURRENT_PYTHON_EXECUTABLE}: Skipping")
    endif()
  endforeach()
endif()

# We always build a rocgdb without python support as fallback.
message(STATUS "Building rocgdb without python support")
add_python_variant("none" "--with-python=no" "")

# Add a single custom target in the main project that depends
# on building ALL rocgdb variants.
add_custom_target(build_all_rocgdb ALL DEPENDS ${ALL_ROCGDB_TARGETS})

################################
# ROCgdb script generation step.
################################

# Since we potentially have multiple versions of python, and each rocgdb
# executable links against a specific version of libpython, we have a
# convenience script that automatically detects the version and fires up
# the most appropriate rocgdb executable. This makes the python selection
# process transparent to the end user.
#
# The script is a POSIX shell script, and there is only something to choose
# between where more than one variant is built.  Windows builds the no-python
# variant alone and installs it as bin/rocgdb.exe, so no chooser goes there.
if(NOT WIN32)
  set(ROCGDB_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/rocgdb")

  # Install the generated script.
  install(PROGRAMS ${ROCGDB_SCRIPT}
          DESTINATION bin
          COMPONENT rocgdb)

  if(BUILD_TESTING)
    # Track for build validation.
    list(APPEND ROCGDB_BUILD_INTEGRITY_FILES "bin/rocgdb")
  endif()
endif()

##################################
# ROCgdb executable patching step.
##################################

# Now add targets for each executable we want to patch post-build.
# These will run after the rocgdb build finishes, and will add the
# appropriate RUNPATH entries so the executables can locate their
# required libraries even if they move around.

# List of all the files in bin we need to patch in RUNPATH's.
# This excludes rocgdb executables as we patch those separately.
set(ROCGDB_EXECS_TO_PATCH)

# roccoremerge is only present in ROCm's branch (not yet in upstream master),
# and comes from binutils, which the Windows build does not configure.  There is
# nothing to patch on Windows in any case: PE has no RUNPATH.
if(NOT ROCGDB_UPSTREAM_BUILD AND NOT WIN32)
  list(APPEND ROCGDB_EXECS_TO_PATCH "bin/roccoremerge")
endif()

foreach(rocgdb_exec_filename ${ROCGDB_EXECS_TO_PATCH})
  set(rocgdb_exec_file "${CMAKE_INSTALL_PREFIX}/${rocgdb_exec_filename}")
  message(STATUS "Registering fixup of ${rocgdb_exec_file}")

  # Patch in the RUNPATH.
  install(CODE "execute_process(
      COMMAND \"${PATCHELF}\" --set-rpath \"\$ORIGIN${ROCGDB_EXTRA_RPATH_SEGMENTS}\"
      \"${rocgdb_exec_file}\"
      COMMAND_ERROR_IS_FATAL ANY
  )"
  COMPONENT rocgdb)
endforeach()

##################################################
# Build integrity and testsuite installation step.
##################################################

if(BUILD_TESTING)
  enable_testing()

  list(APPEND ROCGDB_BUILD_INTEGRITY_FILES ${ROCGDB_EXECS_TO_PATCH})

  # List of all the other files we expect to be in the installation.
  set(ROCGDB_EXPECTED_FILES
    "include/gdb/jit-reader.h"
    "share/info/rocgdb/gdb.info"
    "share/info/rocgdb/annotate.info"
    "share/rocgdb/python/gdb/ptwrite.py"
    "share/rocgdb/python/gdb/__init__.py"
    "share/rocgdb/python/gdb/missing_objfile.py"
    "share/rocgdb/python/gdb/xmethod.py"
    "share/rocgdb/python/gdb/missing_debug.py"
    "share/rocgdb/python/gdb/frames.py"
    "share/rocgdb/python/gdb/FrameIterator.py"
    "share/rocgdb/python/gdb/prompt.py"
    "share/rocgdb/python/gdb/printing.py"
    "share/rocgdb/python/gdb/missing_files.py"
    "share/rocgdb/python/gdb/disassembler.py"
    "share/rocgdb/python/gdb/styling.py"
    "share/rocgdb/python/gdb/function/caller_is.py"
    "share/rocgdb/python/gdb/function/__init__.py"
    "share/rocgdb/python/gdb/function/as_string.py"
    "share/rocgdb/python/gdb/function/rocm_get_scaled_float.py"
    "share/rocgdb/python/gdb/function/strfns.py"
    "share/rocgdb/python/gdb/FrameDecorator.py"
    "share/rocgdb/python/gdb/types.py"
    "share/rocgdb/python/gdb/printer/__init__.py"
    "share/rocgdb/python/gdb/printer/rocm_ocp_mx.py"
    "share/rocgdb/python/gdb/unwinder.py"
    "share/rocgdb/python/gdb/dap/varref.py"
    "share/rocgdb/python/gdb/dap/locations.py"
    "share/rocgdb/python/gdb/dap/threads.py"
    "share/rocgdb/python/gdb/dap/next.py"
    "share/rocgdb/python/gdb/dap/events.py"
    "share/rocgdb/python/gdb/dap/__init__.py"
    "share/rocgdb/python/gdb/dap/launch.py"
    "share/rocgdb/python/gdb/dap/sources.py"
    "share/rocgdb/python/gdb/dap/bt.py"
    "share/rocgdb/python/gdb/dap/state.py"
    "share/rocgdb/python/gdb/dap/typecheck.py"
    "share/rocgdb/python/gdb/dap/globalvars.py"
    "share/rocgdb/python/gdb/dap/server.py"
    "share/rocgdb/python/gdb/dap/evaluate.py"
    "share/rocgdb/python/gdb/dap/scopes.py"
    "share/rocgdb/python/gdb/dap/frames.py"
    "share/rocgdb/python/gdb/dap/startup.py"
    "share/rocgdb/python/gdb/dap/disassemble.py"
    "share/rocgdb/python/gdb/dap/pause.py"
    "share/rocgdb/python/gdb/dap/memory.py"
    "share/rocgdb/python/gdb/dap/completions.py"
    "share/rocgdb/python/gdb/dap/breakpoint.py"
    "share/rocgdb/python/gdb/dap/modules.py"
    "share/rocgdb/python/gdb/dap/io.py"
    "share/rocgdb/python/gdb/command/__init__.py"
    "share/rocgdb/python/gdb/command/pretty_printers.py"
    "share/rocgdb/python/gdb/command/frame_filters.py"
    "share/rocgdb/python/gdb/command/prompt.py"
    "share/rocgdb/python/gdb/command/unwinders.py"
    "share/rocgdb/python/gdb/command/xmethods.py"
    "share/rocgdb/python/gdb/command/type_printers.py"
    "share/rocgdb/python/gdb/command/explore.py"
    "share/rocgdb/python/gdb/command/missing_files.py"
    "share/rocgdb/syscalls/mips-n64-linux.xml"
    "share/rocgdb/syscalls/loongarch-linux.xml"
    "share/rocgdb/syscalls/amd64-linux.xml"
    "share/rocgdb/syscalls/mips-n32-linux.xml"
    "share/rocgdb/syscalls/ppc-linux.xml"
    "share/rocgdb/syscalls/riscv-linux.xml"
    "share/rocgdb/syscalls/sparc-linux.xml"
    "share/rocgdb/syscalls/mips-o32-linux.xml"
    "share/rocgdb/syscalls/aarch64-linux.xml"
    "share/rocgdb/syscalls/ppc64-linux.xml"
    "share/rocgdb/syscalls/netbsd.xml"
    "share/rocgdb/syscalls/sparc64-linux.xml"
    "share/rocgdb/syscalls/arm-linux.xml"
    "share/rocgdb/syscalls/s390-linux.xml"
    "share/rocgdb/syscalls/s390x-linux.xml"
    "share/rocgdb/syscalls/i386-linux.xml"
    "share/rocgdb/syscalls/freebsd.xml"
    "share/rocgdb/syscalls/gdb-syscalls.dtd"
    "share/rocgdb/system-gdbinit/elinos.py"
    "share/rocgdb/system-gdbinit/wrs-linux.py"
    "share/man/man5/rocgdbinit.5"
    "share/man/man1/rocgdb-add-index.1"
    "share/man/man1/rocgdb.1"
  )

  # gdb installs the python runtime files under its data directory only when
  # python support is enabled, so expect them only if some variant actually has
  # it.  That covers Windows too: no variant has python there.
  if(NOT ROCGDB_ANY_PYTHON_VARIANT)
    list(FILTER ROCGDB_EXPECTED_FILES EXCLUDE REGEX "^share/rocgdb/python/")
  endif()

  # Files that only exist where the programs they belong to were built: gcore
  # needs native gcore support, coremerge needs binutils.
  if(NOT WIN32)
    list(APPEND ROCGDB_EXPECTED_FILES "share/man/man1/rocgcore.1")
  endif()

  # AMD-specific files not yet in upstream master.
  if(NOT ROCGDB_UPSTREAM_BUILD)
    list(APPEND ROCGDB_EXPECTED_FILES "share/doc/NOTICES.txt")
    if(NOT WIN32)
      list(APPEND ROCGDB_EXPECTED_FILES "share/man/man1/roccoremerge.1")
    endif()
  endif()

  list(APPEND ROCGDB_BUILD_INTEGRITY_FILES ${ROCGDB_EXPECTED_FILES})
  if(NOT WIN32)
    list(APPEND ROCGDB_BUILD_INTEGRITY_FILES "bin/rocgdb-emulate")
  endif()

  # Check whether files exist through cmake instead of through "sh -c test -f":
  # on Windows nothing puts a POSIX shell on ctest's PATH, and a ctest started
  # from cmd.exe or PowerShell has none.
  set(ROCGDB_FILE_EXISTS_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/check_file_exists.cmake")
  file(WRITE "${ROCGDB_FILE_EXISTS_SCRIPT}"
    "if(NOT EXISTS \"\${ROCGDB_FILE}\")\n"
    "  message(FATAL_ERROR \"Missing expected file: \${ROCGDB_FILE}\")\n"
    "endif()\n")

  # Add build integrity tests to check if we have the files we want.
  message(STATUS "BUILD_TESTING set. Adding rocgdb installation integrity tests.")
  foreach(rocgdb_file ${ROCGDB_BUILD_INTEGRITY_FILES})
    add_test(
      NAME rocgdb_install_validation_${CMAKE_INSTALL_PREFIX}/${rocgdb_file}
      COMMAND ${CMAKE_COMMAND}
        "-DROCGDB_FILE=${CMAKE_INSTALL_PREFIX}/${rocgdb_file}"
        -P "${ROCGDB_FILE_EXISTS_SCRIPT}"
    )
  endforeach()

  # Also install the required testsuite files so we can run tests later on.
  #
  # rocgdb's testsuite does not prebuild executables, so it needs to run on
  # the target machine.

  # Non-executable testsuite files.
  set(ROCGDB_TESTSUITE_FILES
    "config.guess"
    "config.sub"
    "contrib/dg-extract-results.py"
    "gdb/disable-implicit-rules.mk"
    "gdb/silent-rules.mk"
    "include/dwarf2.def"
    "include/dwarf2.h"
  )

  # Executable testsuite files.
  set(ROCGDB_EXECUTABLE_TESTSUITE_FILES
    "contrib/dg-extract-results.sh"
    "install-sh"
  )

  foreach(rocgdb_file ${ROCGDB_TESTSUITE_FILES})
    # Get the directory part of the file.
    get_filename_component(file_dir ${rocgdb_file} DIRECTORY)

    install(FILES ${ROCGDB_SOURCE_DIR}/${rocgdb_file}
      DESTINATION tests/rocgdb/${file_dir}
      COMPONENT tests)
  endforeach()

  foreach(rocgdb_file ${ROCGDB_EXECUTABLE_TESTSUITE_FILES})
    # Get the directory part of the file.
    get_filename_component(file_dir ${rocgdb_file} DIRECTORY)

    install(PROGRAMS ${ROCGDB_SOURCE_DIR}/${rocgdb_file}
      DESTINATION tests/rocgdb/${file_dir}
      COMPONENT tests)
  endforeach()

  # Testsuite directories.
  set(ROCGDB_TESTSUITE_DIRS
    "gdb/contrib"
    "gdb/features"
    "gdb/testsuite"
  )

  foreach(rocgdb_testsuite_dir ${ROCGDB_TESTSUITE_DIRS})
    install(DIRECTORY ${ROCGDB_SOURCE_DIR}/${rocgdb_testsuite_dir}
      DESTINATION tests/rocgdb/gdb
      COMPONENT tests
      USE_SOURCE_PERMISSIONS)
  endforeach()

  # Install the CI wrapper that invokes the rocgdb testsuite, the emulator
  # module it imports for --emulate, and rocgdb-emulate, which runs rocgdb
  # inside a mirage session. All are taken from the source tree. If any is
  # missing and ROCGDB_DOWNLOAD_CI_SCRIPT is enabled, all are fetched from the
  # same amd-staging commit so they always match.
  set(ROCGDB_CI_SCRIPTS test_rocgdb.py rocjitsu_emulator.py rocgdb_emulate.py)
  set(ROCGDB_CI_SCRIPTS_DIR "${ROCGDB_SOURCE_DIR}/.github/scripts")
  set(missing_ci_scripts)
  foreach(script IN LISTS ROCGDB_CI_SCRIPTS)
    if(NOT EXISTS "${ROCGDB_CI_SCRIPTS_DIR}/${script}")
      list(APPEND missing_ci_scripts "${script}")
    endif()
  endforeach()

  if(missing_ci_scripts)
    list(JOIN missing_ci_scripts ", " missing_ci_scripts_list)
    if(NOT ROCGDB_DOWNLOAD_CI_SCRIPT)
      message(FATAL_ERROR "${missing_ci_scripts_list} not found in ${ROCGDB_CI_SCRIPTS_DIR} and THEROCK_ROCGDB_DOWNLOAD_CI_SCRIPT is OFF. Re-configure with -DTHEROCK_ROCGDB_DOWNLOAD_CI_SCRIPT=ON to fetch them from amd-staging.")
    endif()
    message(STATUS "${missing_ci_scripts_list} missing from the source tree")
    set(ROCGDB_CI_SCRIPTS_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    _rocgdb_download_ci_scripts("${ROCGDB_CI_SCRIPTS_DIR}" ${ROCGDB_CI_SCRIPTS})
  endif()

  install(
    PROGRAMS ${ROCGDB_CI_SCRIPTS_DIR}/test_rocgdb.py
    DESTINATION tests/rocgdb
    COMPONENT tests
  )

  install(
    FILES ${ROCGDB_CI_SCRIPTS_DIR}/rocjitsu_emulator.py
    DESTINATION tests/rocgdb
    COMPONENT tests
  )

  # Installed next to rocgdb, without the .py extension.  Not on Windows, where
  # the emulation stack it runs rocgdb under (mirage) is not built.
  if(NOT WIN32)
    install(
      PROGRAMS ${ROCGDB_CI_SCRIPTS_DIR}/rocgdb_emulate.py
      DESTINATION bin
      RENAME rocgdb-emulate
      COMPONENT rocgdb
    )
  endif()

  # Install the optional xfails ignore list, if present in the sources.
  if(EXISTS "${ROCGDB_SOURCE_DIR}/.github/scripts/rocgdb_ignore_list.json")
    install(
      FILES ${ROCGDB_SOURCE_DIR}/.github/scripts/rocgdb_ignore_list.json
      DESTINATION tests/rocgdb
      COMPONENT tests
    )
  else()
    message(WARNING "rocgdb tests: xfails ignore list not found at "
      "${ROCGDB_SOURCE_DIR}/.github/scripts/rocgdb_ignore_list.json; skipping installation.")
  endif()
else()
  message(STATUS "rocgdb tests: BUILD_TESTING not set. Skipping installation integrity tests.")
endif()
