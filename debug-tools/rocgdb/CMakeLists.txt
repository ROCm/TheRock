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
fetch_library_prefix(CMAKE_PREFIX_PATH "gmp" GMP_PREFIX)
fetch_library_prefix(CMAKE_PREFIX_PATH "mpfr" MPFR_PREFIX)
fetch_library_prefix(CMAKE_PREFIX_PATH "expat" EXPAT_PREFIX)
fetch_library_prefix(CMAKE_PREFIX_PATH "ncurses" NCURSES_PREFIX)

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
set(ROCGDB_HTML_DIR "\\\${prefix}/share/html")
set(ROCGDB_PDF_DIR "\\\${prefix}/share/doc")
set(ROCGDB_INFO_DIR "\\\${prefix}/share/info")
set(ROCGDB_SEPARATE_DEBUG_INFO_DIR "\\\${prefix}/lib/debug:/usr/lib/debug")
set(ROCGDB_DATA_DIR "\\\${prefix}/share")
set(ROCGDB_TARGETS "x86_64-linux-gnu,amdgcn-amd-amdhsa")

# If the version is not set, set it to a generic value.
if (NOT ROCM_VERSION)
  set(ROCM_BUILD_ID "ROCm development")
else()
  set(ROCM_BUILD_ID "ROCm-${ROCM_VERSION}")
endif()

# List to accumulate all unique build targets.
set(ALL_GDB_TARGETS)

# Adds a rocgdb build target for a python variant with explicit additional configure flags.
# The special minor_version "none" indicates that the build is without python support.
macro(add_python_variant minor_version configure_flags)
  # Set the name transformation for the rocgdb executable and rocgcore script.
  set(ROCGDB_PROGRAM_TRANSFORM_STR "s/^rocgdb$/rocgdb-py${minor_version}/")

  # Define a unique name for the build directory. The install prefix will be
  # the same, as all the files minus the rocgdb executable will be the same,
  # no matter which version of python we have. We will rename the rocgdb
  # executables later.
  set(GDB_BUILD_DIR_VER "${CMAKE_CURRENT_BINARY_DIR}/rocgdb-build-py${minor_version}")
  set(GDB_TARGET_NAME "rocgdb-py${minor_version}")

  add_custom_target("${GDB_TARGET_NAME}"
    ALL
    VERBATIM
    COMMENT "rocgdb (python ${CURRENT_PYTHON_EXECUTABLE})"
    # Create the build directory
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GDB_BUILD_DIR_VER}"

    # Configure
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${GDB_BUILD_DIR_VER}"
      ${CMAKE_COMMAND} -E env
        "CC=${CUSTOM_CC}"
        "CXX=${CUSTOM_CXX}"
        "PKG_CONFIG_PATH=${PKG_CONFIG_PATH_ENV_VAR}"
        "CFLAGS=${CUSTOM_C_FLAGS}"
        "CXXFLAGS=${CUSTOM_CXX_FLAGS}"
        "LDFLAGS=${LDFLAGS}"
      --
      # The actual configure command.
      ${CMAKE_CURRENT_SOURCE_DIR}/source/configure
        "--prefix=${CMAKE_INSTALL_PREFIX}"
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
      ${CMAKE_COMMAND} -E chdir "${GDB_BUILD_DIR_VER}"
      ${MAKE_EXECUTABLE} -j "${PAR_JOBS}" V=1

    # Install gdb and gdb docs.
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${GDB_BUILD_DIR_VER}"
      ${MAKE_EXECUTABLE} -C gdb install install-pdf install-html
    # Install binutils for coremerge and coremerge manpage.
    COMMAND
      ${CMAKE_COMMAND} -E chdir "${GDB_BUILD_DIR_VER}"
      ${MAKE_EXECUTABLE} -C binutils install
    # Add in the AMD licences file.
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/share/doc
    COMMAND
      ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/source/gdb/NOTICES.txt ${CMAKE_INSTALL_PREFIX}/share/doc/
    # Rename rocgcore to rocgcore-py<python version>.
    COMMAND
      ${CMAKE_COMMAND} -E rename ${CMAKE_INSTALL_PREFIX}/bin/rocgcore ${CMAKE_INSTALL_PREFIX}/bin/rocgcore-py${minor_version}
  )

  list(APPEND ALL_GDB_TARGETS ${GDB_TARGET_NAME})

  # Record the available python versions.
  list(APPEND ALL_AVAILABLE_PYTHON_VERSIONS ${minor_version})
endmacro()

# Determine python versions either from the explicit list of dist python executables
# or from the global Python3_EXECUTABLE.
if(SHARED_PYTHON_EXECUTABLES)
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
    message(STATUS "Building for python ${CURRENT_PYTHON_EXECUTABLE}: Minor=${CURRENT_PYTHON_VERSION}, Full=${CURRENT_PYTHON_FULL_VERSION}")
    message(STATUS "-> Found Python ${CURRENT_PYTHON_FULL_VERSION} at ${CURRENT_PYTHON_EXECUTABLE}")
    add_python_variant("${CURRENT_PYTHON_VERSION}" "--with-python=${CURRENT_PYTHON_EXECUTABLE}")
  endforeach()
else()
  # We were not told to build with python support.
  add_python_variant("none" "--disable-python")
endif()

# Add a single custom target in the main project that depends
# on building ALL GDB versions.
add_custom_target(build_all_gdb ALL DEPENDS ${ALL_GDB_TARGETS})

# Print the base install location.
message(STATUS "rocgdb will be installed under ${CMAKE_INSTALL_PREFIX}")

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
install(PROGRAMS "${ROCGDB_SCRIPT}"
        DESTINATION bin
        COMPONENT tests)

##################################
# ROCgdb executable patching step.
##################################

# Now add targets for each executable we want to patch post-build.
# These will run after the rocgdb build finishes, and will add the
# appropriate RUNPATH entries so the executables can locate their
# required libraries even if they move around.

# List of all the files in bin we need to patch in RUNPATH's.
set(ROCGDB_EXECS_TO_PATCH
  "rocaddr2line"
  "rocar"
  "rocc++filt"
  "roccoremerge"
  "rocnm"
  "rocobjcopy"
  "rocobjdump"
  "rocranlib"
  "rocreadelf"
  "rocsize"
  "rocstrings"
  "rocstrip"
)

# Add the python-specific rocgdb executables.
foreach(python_version ${ALL_AVAILABLE_PYTHON_VERSIONS})
  list(APPEND ROCGDB_EXECS_TO_PATCH "rocgdb-py${python_version}")
endforeach()

foreach(rocgdb_exec_filename ${ROCGDB_EXECS_TO_PATCH})
  set(rocgdb_exec_file "${CMAKE_INSTALL_PREFIX}/bin/${rocgdb_exec_filename}")
  message(STATUS "Registering fixup of ${rocgdb_exec_file}")

  # Patch in the RUNPATH.
  install(CODE "execute_process(
      COMMAND \"${PATCHELF}\" --set-rpath \"\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../lib/rocm_sysdeps/lib\"
      \"\${CMAKE_INSTALL_PREFIX}/bin/${rocgdb_exec_filename}\"
      COMMAND_ERROR_IS_FATAL ANY
  )"
  COMPONENT rocgdb)
endforeach()

##################################################
# Build integrity and testsuite installation step.
##################################################

if(BUILD_TESTING)
  enable_testing()

  # List of all the files in x86_64-pc-linux-gnu/bin.
  #
  # NOTE: We don't need to patch the files in ROCGDB_TARGET_EXECS as
  # those are hardlinks that point to similar files in ROCGDB_EXECS_TO_PATCH.
  set(ROCGDB_TARGET_EXECS
    "ar"
    "nm"
    "objcopy"
    "objdump"
    "ranlib"
    "readelf"
    "strip"
  )

  # List of all the other files we expect to be in the installation.
  set(ROCGDB_EXPECTED_FILES
    "include/gdb/jit-reader.h"
    "share/info/gdb.info"
    "share/info/annotate.info"
    "share/info/binutils.info"
    "share/info/dir"
    "share/python/gdb/ptwrite.py"
    "share/python/gdb/__init__.py"
    "share/python/gdb/missing_objfile.py"
    "share/python/gdb/xmethod.py"
    "share/python/gdb/missing_debug.py"
    "share/python/gdb/frames.py"
    "share/python/gdb/FrameIterator.py"
    "share/python/gdb/prompt.py"
    "share/python/gdb/printing.py"
    "share/python/gdb/missing_files.py"
    "share/python/gdb/disassembler.py"
    "share/python/gdb/styling.py"
    "share/python/gdb/function/caller_is.py"
    "share/python/gdb/function/__init__.py"
    "share/python/gdb/function/as_string.py"
    "share/python/gdb/function/rocm_get_scaled_float.py"
    "share/python/gdb/function/strfns.py"
    "share/python/gdb/FrameDecorator.py"
    "share/python/gdb/types.py"
    "share/python/gdb/printer/__init__.py"
    "share/python/gdb/printer/rocm_ocp_mx.py"
    "share/python/gdb/unwinder.py"
    "share/python/gdb/dap/varref.py"
    "share/python/gdb/dap/locations.py"
    "share/python/gdb/dap/threads.py"
    "share/python/gdb/dap/next.py"
    "share/python/gdb/dap/events.py"
    "share/python/gdb/dap/__init__.py"
    "share/python/gdb/dap/launch.py"
    "share/python/gdb/dap/sources.py"
    "share/python/gdb/dap/bt.py"
    "share/python/gdb/dap/state.py"
    "share/python/gdb/dap/typecheck.py"
    "share/python/gdb/dap/globalvars.py"
    "share/python/gdb/dap/server.py"
    "share/python/gdb/dap/evaluate.py"
    "share/python/gdb/dap/scopes.py"
    "share/python/gdb/dap/frames.py"
    "share/python/gdb/dap/startup.py"
    "share/python/gdb/dap/disassemble.py"
    "share/python/gdb/dap/pause.py"
    "share/python/gdb/dap/memory.py"
    "share/python/gdb/dap/completions.py"
    "share/python/gdb/dap/breakpoint.py"
    "share/python/gdb/dap/modules.py"
    "share/python/gdb/dap/io.py"
    "share/python/gdb/command/__init__.py"
    "share/python/gdb/command/pretty_printers.py"
    "share/python/gdb/command/frame_filters.py"
    "share/python/gdb/command/prompt.py"
    "share/python/gdb/command/unwinders.py"
    "share/python/gdb/command/xmethods.py"
    "share/python/gdb/command/type_printers.py"
    "share/python/gdb/command/explore.py"
    "share/python/gdb/command/missing_files.py"
    "share/syscalls/mips-n64-linux.xml"
    "share/syscalls/loongarch-linux.xml"
    "share/syscalls/amd64-linux.xml"
    "share/syscalls/mips-n32-linux.xml"
    "share/syscalls/ppc-linux.xml"
    "share/syscalls/riscv-linux.xml"
    "share/syscalls/sparc-linux.xml"
    "share/syscalls/mips-o32-linux.xml"
    "share/syscalls/aarch64-linux.xml"
    "share/syscalls/ppc64-linux.xml"
    "share/syscalls/netbsd.xml"
    "share/syscalls/sparc64-linux.xml"
    "share/syscalls/arm-linux.xml"
    "share/syscalls/s390-linux.xml"
    "share/syscalls/s390x-linux.xml"
    "share/syscalls/i386-linux.xml"
    "share/syscalls/freebsd.xml"
    "share/syscalls/gdb-syscalls.dtd"
    "share/system-gdbinit/elinos.py"
    "share/system-gdbinit/wrs-linux.py"
    "share/man/man5/rocgdbinit.5"
    "share/man/man1/rocgdb-add-index.1"
    "share/man/man1/rocdlltool.1"
    "share/man/man1/rocsize.1"
    "share/man/man1/rocobjdump.1"
    "share/man/man1/rocstrings.1"
    "share/man/man1/rocc++filt.1"
    "share/man/man1/rocgcore.1"
    "share/man/man1/rocgdb.1"
    "share/man/man1/rocar.1"
    "share/man/man1/rocnm.1"
    "share/man/man1/rocelfedit.1"
    "share/man/man1/rocwindmc.1"
    "share/man/man1/rocgstack.1"
    "share/man/man1/roccoremerge.1"
    "share/man/man1/rocobjcopy.1"
    "share/man/man1/rocreadelf.1"
    "share/man/man1/rocstrip.1"
    "share/man/man1/rocaddr2line.1"
    "share/man/man1/rocranlib.1"
    "share/man/man1/rocwindres.1"
    "share/man/man1/rocgdbserver.1"
  )

  # Add the python-specific rocgcore scripts. We add this later as otherwise
  # we'd attempt to call patchelf for a script, and that would fail.
  foreach(python_version ${ALL_AVAILABLE_PYTHON_VERSIONS})
    list(APPEND ROCGDB_EXECS_TO_PATCH "rocgcore-py${python_version}")
  endforeach()

  # Add build integrity tests to check if we have the files we want.
  message(STATUS "rocgdb tests: BUILD_TESTING set. Adding installation integrity tests.")
  # Check if we've built all the files we wanted to build.
  foreach(rocgdb_file ${ROCGDB_EXECS_TO_PATCH})
    add_test(
      NAME rocgdb_install_validation_${CMAKE_INSTALL_PREFIX}/bin/${rocgdb_file}
      COMMAND ${CMAKE_COMMAND} -E env
        sh -c "test -f ${CMAKE_INSTALL_PREFIX}/bin/${rocgdb_file}"
    )
  endforeach()

  foreach(rocgdb_file ${ROCGDB_TARGET_EXECS})
    add_test(
      NAME rocgdb_install_validation_${CMAKE_INSTALL_PREFIX}/x86_64-pc-linux-gnu/bin/${rocgdb_file}
      COMMAND ${CMAKE_COMMAND} -E env
        sh -c "test -f ${CMAKE_INSTALL_PREFIX}/x86_64-pc-linux-gnu/bin/${rocgdb_file}"
    )
  endforeach()

  foreach(rocgdb_file ${ROCGDB_EXPECTED_FILES})
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
