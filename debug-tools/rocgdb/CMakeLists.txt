## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

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
# each entry supports linking against libpython.
#
# Example: SHARED_PYTHON_EXECUTABLES="/opt/python-shared/cp310-cp310/bin/python3"


cmake_minimum_required(VERSION 3.25)
project(rocgdb C CXX)

include(ExternalProject)
include(ProcessorCount)

# Function to extract a single path that matches a given regex pattern
# from a list of paths.
#
# Usage:
#   fetch_library_prefix(<input_path_list> <pattern> <output_var>)
#
# Example:
#   fetch_library_prefix(CMAKE_PREFIX_PATH "gmp" GMP_PREFIX)
#
function(fetch_library_prefix input_path_list pattern output_var)
  # Access the list contents using the input variable name
  set(_input_list "${${input_path_list}}")

  # Filter the list using the provided pattern
  list(FILTER _input_list INCLUDE REGEX "${pattern}")

  # Check if a match was found.
  if(_input_list)
    # Store the first match in the output variable provided by the caller.
    list(GET _input_list 0 _result_path)
    set(${output_var} "${_result_path}" PARENT_SCOPE)
    message(STATUS "Extracted path for pattern '${pattern}': ${_result_path}")
  else()
    # No match found, issue a fatal error.
    message(SEND_ERROR "Did not find a path containing '${pattern}' in ${input_list_var_name}.")
  endif()
endfunction()

# By default install the ROCgdb testsuite files.
option(ENABLE_TESTS "Enable installation of ROCgdb testsuite files" OFF)

# Find standard build tools.
find_program(MAKE_EXECUTABLE NAMES make REQUIRED)

ProcessorCount(PAR_JOBS)

# Configure build type flags.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Standard flags are handled by CMAKE_C_FLAGS, CMAKE_C_FLAGS_DEBUG, etc.
set(CUSTOM_C_FLAGS)
set(CUSTOM_CXX_FLAGS)

# Set custom flags for Debug builds.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(CUSTOM_C_FLAGS "-O0 -g3")
  set(CUSTOM_CXX_FLAGS "-O0 -g3")
endif()

# Compiler commands to chain to autotools.
set(CUSTOM_CC "${CMAKE_C_COMPILER}")
if(CMAKE_C_COMPILER_LAUNCHER)
  string(PREPEND CUSTOM_CC "${CMAKE_C_COMPILER_LAUNCHER} ")
endif()
set(CUSTOM_CXX "${CMAKE_CXX_COMPILER}")
if(CMAKE_CXX_COMPILER_LAUNCHER)
  string(PREPEND CUSTOM_CXX "${CMAKE_CXX_COMPILER_LAUNCHER} ")
endif()

message(STATUS "Build type set to: ${CMAKE_BUILD_TYPE}")

# Fetch the install prefixes for the library and executables rocgdb
# depends on.
fetch_library_prefix(CMAKE_PREFIX_PATH "/gmp/build/stage" GMP_PREFIX)
fetch_library_prefix(CMAKE_PREFIX_PATH "/mpfr/build/stage" MPFR_PREFIX)
fetch_library_prefix(CMAKE_PREFIX_PATH "/expat/build/stage" EXPAT_PREFIX)
fetch_library_prefix(CMAKE_PREFIX_PATH "/ncurses/build/stage" NCURSES_PREFIX)

# We need to explicitly pass ncurses' include path and set the rpath for
# the library. The reason we need to explicitly pass rpath is because gdb's
# configure assumes ncurses is always picked up from the default system
# location, or that the library is always within reach.
#
# Since we're building our own ncurses and it is in a non-standard location,
# the rpath assures the configure tests properly locate the library.
set(NCURSES_RPATH "-Wl,-rpath,${NCURSES_PREFIX}/lib")
set(NCURSES_INCLUDE "-I${NCURSES_PREFIX}/include")

# Adjust the compilation flags and LDFLAGS.
set(CUSTOM_C_FLAGS "${CUSTOM_C_FLAGS} ${NCURSES_INCLUDE}")
set(CUSTOM_CXX_FLAGS "${CUSTOM_CXX_FLAGS} ${NCURSES_INCLUDE}")
set(LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NCURSES_RPATH} -Wl,--enable-new-dtags")

# Replace all semicolons (the CMake separator) with colons (the ENV separator).
# We need this to pass it on to rocgdb's configure.
string(REPLACE ";" ":" PKG_CONFIG_PATH_ENV_VAR "${THEROCK_PKG_CONFIG_DIRS}")

# Set some global variables for the configure/build.
set(ROCGDB_BUG_URL "https://github.com/ROCm-Developer-Tools/ROCgdb/issues")
set(ROCGDB_SEPARATE_DEBUG_INFO_DIR "\\\${prefix}/lib/debug:/usr/lib/debug")
set(ROCGDB_DATA_DIR "\\\${prefix}/share/rocgdb")
set(ROCGDB_TARGETS "x86_64-linux-gnu,amdgcn-amd-amdhsa")

# If the version is not set, set it to a generic value.
if (NOT ROCM_VERSION)
  set(ROCM_BUILD_ID "ROCm development")
else()
  set(ROCM_BUILD_ID "ROCm-${ROCM_VERSION}")
endif()

# List to accumulate all unique build targets.
set(ALL_ROCGDB_TARGETS)

# Adds a rocgdb build target for a python variant with explicit additional configure flags.
# The special minor_version "none" indicates that the build is without python support.
#
# For variants with minor_version other than "none", libpython_dir must contain
# the path to the shared libpython rocgdb should link against.
macro(add_python_variant minor_version configure_flags libpython_dir)
  # Set the name transformation for the rocgdb executable and rocgcore script.
  set(ROCGDB_PROGRAM_TRANSFORM_STR "s/^rocgdb$/rocgdb-py${minor_version}/")

  # Define a unique name for the build directory.
  set(ROCGDB_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocgdb-build-py${minor_version}")
  # Define a unique name for the intermediate install directory.
  set(ROCGDB_INTERMEDIATE_INSTALL_DIR "${ROCGDB_BUILD_DIR}-install")
  set(ROCGDB_TARGET_NAME "rocgdb-py${minor_version}")

  # If we have a valid libpython_dir, adjust the LDFLAGS for this build
  # variant and add an extra RPATH entry for post-install patching.
  set(ALL_LDFLAGS ${LDFLAGS})
  set(LIBPYTHON_RPATH "")
  if(NOT "${libpython_dir}" STREQUAL "")
    set(LIBPYTHON_LDFLAGS "-L${libpython_dir} -Wl,-rpath,${libpython_dir}")
    string(APPEND ALL_LDFLAGS " ${LIBPYTHON_LDFLAGS}")
    set(LIBPYTHON_RPATH ":${libpython_dir}")

    message(STATUS "-> Extra LDFLAGS: ${LIBPYTHON_LDFLAGS}")
    message(STATUS "-> Extra RPATH: ${LIBPYTHON_RPATH}")
  endif()

  # These directories are adjusted here so we don't have to escape
  # ${prefix} and make it survive across multiple expansions from shells
  # cmake and libtool.
  set(ROCGDB_HTML_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/html")
  set(ROCGDB_PDF_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/doc")
  set(ROCGDB_INFO_DIR "${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/info/rocgdb")

  add_custom_target("${ROCGDB_TARGET_NAME}"
    ALL
    VERBATIM
    COMMENT "rocgdb (python ${CURRENT_PYTHON_EXECUTABLE})"
    # Create the build directory
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ROCGDB_BUILD_DIR}"

    # Configure
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${ROCGDB_BUILD_DIR}"
      ${CMAKE_COMMAND} -E env
        "CC=${CUSTOM_CC}"
        "CXX=${CUSTOM_CXX}"
        "PKG_CONFIG_PATH=${PKG_CONFIG_PATH_ENV_VAR}"
        "CFLAGS=${CUSTOM_C_FLAGS}"
        "CXXFLAGS=${CUSTOM_CXX_FLAGS}"
        "LDFLAGS=${ALL_LDFLAGS}"
      --
      # The actual configure command.
      ${CMAKE_CURRENT_SOURCE_DIR}/source/configure
        "--prefix=${ROCGDB_INTERMEDIATE_INSTALL_DIR}"
        --program-prefix=roc
        "--program-transform-name=${ROCGDB_PROGRAM_TRANSFORM_STR}"

        --enable-64-bit-bfd
        "--enable-targets=${ROCGDB_TARGETS}"
        --enable-tui

        "--htmldir=${ROCGDB_HTML_DIR}"
        "--pdfdir=${ROCGDB_PDF_DIR}"
        "--infodir=${ROCGDB_INFO_DIR}"

        --with-amd-dbgapi
        "--with-bugurl=${ROCGDB_BUG_URL}"
        "--with-expat-prefix=${EXPAT_PREFIX}"
        "--with-gdb-datadir=${ROCGDB_DATA_DIR}"
        "--with-gmp=${GMP_PREFIX}"
        --with-lzma
        "--with-mpfr=${MPFR_PREFIX}"
        "--with-pkgversion=${ROCM_BUILD_ID}"
        "--with-separate-debug-dir=${ROCGDB_SEPARATE_DEBUG_INFO_DIR}"
        --with-system-zlib
        --with-zstd

        --disable-gas
        --disable-gdbserver
        --disable-gdbtk
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

    # Build
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${ROCGDB_BUILD_DIR}"
      ${MAKE_EXECUTABLE} -s -j "${PAR_JOBS}"
    # Install gdb.
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${ROCGDB_BUILD_DIR}"
      ${MAKE_EXECUTABLE} -s -C gdb install
    # Install gdb docs.
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${ROCGDB_BUILD_DIR}"
      ${MAKE_EXECUTABLE} -s -C gdb install-html
    # Install binutils for coremerge and coremerge manpage.
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${ROCGDB_BUILD_DIR}"
      ${MAKE_EXECUTABLE} -s -C binutils install
    # Add in the AMD licences file.
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/doc
    COMMAND
      ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/source/gdb/NOTICES.txt ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/share/doc/
    # Rename rocgcore to rocgcore-py<python version>.
    COMMAND
      ${CMAKE_COMMAND} -E rename ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/bin/rocgcore ${ROCGDB_INTERMEDIATE_INSTALL_DIR}/bin/rocgcore-py${minor_version}
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
    "bin/roccoremerge"
    "bin/rocgcore-py${minor_version}"
    "bin/rocgdb-py${minor_version}"
  )

  # Misc files.
  set(ROCGDB_MISC_FILES_TO_INSTALL
    "share/info/rocgdb/annotate.info"
    "share/info/rocgdb/gdb.info"
    "share/man/man1/roccoremerge.1"
    "share/man/man1/rocgdb.1"
    "share/man/man1/rocgcore.1"
    "share/man/man1/rocgdb-add-index.1"
  )

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

  # Register RPATH patching for this rocgdb variant.
  message(STATUS "Registering fixup of ${CMAKE_INSTALL_PREFIX}/bin/rocgdb-py${minor_version}")
  install(CODE "execute_process(
      COMMAND \"${PATCHELF}\" --set-rpath \"\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rocm_sysdeps/lib${LIBPYTHON_RPATH}\"
      \"${CMAKE_INSTALL_PREFIX}/bin/rocgdb-py${minor_version}\"
      COMMAND_ERROR_IS_FATAL ANY
    )"
  COMPONENT rocgdb)

  if(BUILD_TESTING)
    # Track for build validation.
    list(APPEND ROCGDB_BUILD_INTEGRITY_FILES "bin/rocgdb-py${minor_version}")
    list(APPEND ROCGDB_BUILD_INTEGRITY_FILES "bin/rocgcore-py${minor_version}")
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
add_python_variant("none" "--disable-python" "")

# Add a single custom target in the main project that depends
# on building ALL rocgdb variants.
add_custom_target(build_all_rocgdb ALL DEPENDS ${ALL_ROCGDB_TARGETS})

################################
# ROCgdb script generation step.
################################

# Since we potentially have multiple versions of python, and each rocgdb
# executable links against a specific version of libpython, we have to
# generate a convenience script that automatically detects the version and
# fires up the most appropriate rocgdb executable. This makes the python
# selection process transparent to the end user.
set(ROCGDB_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/rocgdb")

# We use [=[ to capture the exact script content.
set(ROCGDB_SCRIPT_CONTENT [=[#!/usr/bin/env bash

## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# Find the active python version
python_version=$(python3 --version 2>&1 | sed -n 's/^Python \([0-9]\+\.[0-9]\+\).*$/\1/p' )
GDB_BIN_DIR="$(dirname "$(realpath -e "${BASH_SOURCE[0]}")")"

# Set the ncurses TERMINFO lookup directory if it isn't set already.
# This is used to locate our own terminfo database in a relative path, since
# ncurses doesn't support relative path lookups as a configure option.
#
# ROCgdb TUI mode requires this information.
if [ -z "$TERMINFO" ]; then
    export TERMINFO="$GDB_BIN_DIR/../lib/rocm_sysdeps/share/terminfo"
fi

# Create a rocgdb command with the active python version.
gdb_command="rocgdb-py$python_version"
if ! [ -f "$GDB_BIN_DIR/$gdb_command" ]; then
    # If we were built without python support, then try the -pynone binary.
    gdb_command="rocgdb-pynone"
fi
if [ -f "$GDB_BIN_DIR/$gdb_command" ]; then
    # If the command works (returns 0), exec the rocgdb-py<version> binary.
    "$GDB_BIN_DIR/$gdb_command" --version > /dev/null 2>&1
    if [ $? == 0 ]; then
        exec "$GDB_BIN_DIR/$gdb_command" "$@"
    fi
fi

echo "rocgdb execution failed. Possible unsupported python version ${python_version} in the system."
]=])

# Generate the script file.
file(GENERATE OUTPUT "${ROCGDB_SCRIPT}" CONTENT "${ROCGDB_SCRIPT_CONTENT}")

# Install the generated script.
install(PROGRAMS ${ROCGDB_SCRIPT}
        DESTINATION bin
        COMPONENT rocgdb)

if(BUILD_TESTING)
  # Track for build validation.
  list(APPEND ROCGDB_BUILD_INTEGRITY_FILES "bin/rocgdb")
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
set(ROCGDB_EXECS_TO_PATCH
  "bin/roccoremerge"
)

foreach(rocgdb_exec_filename ${ROCGDB_EXECS_TO_PATCH})
  set(rocgdb_exec_file "${CMAKE_INSTALL_PREFIX}/${rocgdb_exec_filename}")
  message(STATUS "Registering fixup of ${rocgdb_exec_file}")

  # Patch in the RUNPATH.
  install(CODE "execute_process(
      COMMAND \"${PATCHELF}\" --set-rpath \"\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rocm_sysdeps/lib\"
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
    "share/man/man1/rocgcore.1"
    "share/man/man1/rocgdb.1"
    "share/man/man1/roccoremerge.1"
  )

  list(APPEND ROCGDB_BUILD_INTEGRITY_FILES ${ROCGDB_EXPECTED_FILES})

  # Add build integrity tests to check if we have the files we want.
  message(STATUS "BUILD_TESTING set. Adding rocgdb installation integrity tests.")
  foreach(rocgdb_file ${ROCGDB_BUILD_INTEGRITY_FILES})
    add_test(
      NAME rocgdb_install_validation_${CMAKE_INSTALL_PREFIX}/${rocgdb_file}
      COMMAND ${CMAKE_COMMAND} -E env
        sh -c "test -f ${CMAKE_INSTALL_PREFIX}/${rocgdb_file}"
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

    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/${rocgdb_file}
      DESTINATION tests/rocgdb/${file_dir}
      COMPONENT tests)
  endforeach()

  foreach(rocgdb_file ${ROCGDB_EXECUTABLE_TESTSUITE_FILES})
    # Get the directory part of the file.
    get_filename_component(file_dir ${rocgdb_file} DIRECTORY)

    install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/source/${rocgdb_file}
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
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/source/${rocgdb_testsuite_dir}
      DESTINATION tests/rocgdb/gdb
      COMPONENT tests
      USE_SOURCE_PERMISSIONS)
  endforeach()
else()
  message(STATUS "rocgdb tests: BUILD_TESTING not set. Skipping installation integrity tests.")
endif()
