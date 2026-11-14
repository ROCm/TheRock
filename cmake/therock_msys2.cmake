# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_msys2.cmake
# Helpers for building autotools-based components on Windows through an MSYS2
# UCRT64 login shell.  Several bundled sysdeps and rocgdb have GNU autotools
# build systems that need a POSIX environment (/bin/sh, make) and the ucrt64
# gcc/g++ toolchain.  Native MSVC cannot drive those, so the
# configure/make/install steps are shelled out to MSYS2 while the rest of
# TheRock keeps building natively.
#
# See "Install tools" in docs/development/windows_support.md for setting up
# MSYS2.

if(WIN32)
  # Root of the MSYS2 installation to build under.  Defaults to the standard
  # msys2.org installer location.  Override at configure time if MSYS2 lives
  # elsewhere.  A sub-project's own configure gets the super-project's value
  # through therock_msys2_host_gnu_toolchain_args() below.
  #
  # Set it when a build tree is first configured.  The MSYS2-built
  # sub-projects' own CMake caches keep the gcc they were first configured
  # with: after a change they go on using the old one while it exists, and
  # fail once it is gone.  Clear their build/build and stamp/ directories to
  # move them to another MSYS2.
  set(THEROCK_MSYS2_ROOT "C:/msys64"
    CACHE PATH "Root of the MSYS2 installation used to build autotools components on Windows")
  set(THEROCK_MSYS2_BASH "${THEROCK_MSYS2_ROOT}/usr/bin/bash.exe")
endif()

# therock_msys2_host_gnu_toolchain_args(<target_name> <out_var>)
#
# Set <out_var> to the therock_cmake_subproject_declare() arguments for a
# sub-project built by MSYS2's ucrt64 gcc on Windows, under the host-gnu
# toolchain: EXTERNAL_C_COMPILER and EXTERNAL_CXX_COMPILER naming that gcc and
# g++, a CMAKE_ARGS entry handing the sub-project's own configure
# THEROCK_MSYS2_ROOT, and this file as an FPRINT_FILE_GLOBS and an
# EXTRA_DEPENDS entry.  Splice <out_var> into the declare call alongside the
# sub-project's own arguments, which add to these.  Call it where such a
# sub-project is declared, and only there.  Sub-projects are declared only when
# enabled, so a build that enables none of them does not need MSYS2 at all.
#
# The caller names COMPILER_TOOLCHAIN host-gnu itself, as a literal keyword in
# the declare call (its value may come from a variable).  build_tools' static
# consumer-graph analyzer groups a declare call's arguments by the keywords it
# can see before expanding any variable, so a COMPILER_TOOLCHAIN hidden in
# <out_var> would be invisible to it.
#
# Fails configure if MSYS2's bash or its ucrt64 gcc or g++ is missing, naming
# <target_name>, so that a missing or half-populated MSYS2 is reported up front
# instead of deep inside the sub-project's build.
function(therock_msys2_host_gnu_toolchain_args target_name out_var)
  # The compilers live under the ucrt64 subsystem instead of usr/bin, so an
  # MSYS2 with no toolchain installed has bash but not gcc.
  set(_gcc "${THEROCK_MSYS2_ROOT}/ucrt64/bin/gcc.exe")
  set(_gxx "${THEROCK_MSYS2_ROOT}/ucrt64/bin/g++.exe")
  foreach(_required IN ITEMS "${THEROCK_MSYS2_BASH}" "${_gcc}" "${_gxx}")
    if(NOT EXISTS "${_required}")
      message(FATAL_ERROR
        "${target_name} is built by MSYS2's UCRT64 gcc on Windows, but "
        "${_required} does not exist.  Run build_tools/setup_msys2.ps1, which "
        "reports what is missing and sets MSYS2 up with -Install, set "
        "-DTHEROCK_MSYS2_ROOT to an existing MSYS2 install, or turn off the "
        "feature that enables ${target_name}.")
    endif()
  endforeach()
  # The sub-project's configure includes this file too, through
  # therock_autotools.cmake, to find MSYS2's bash.  Without the forward it
  # would fall back to the C:/msys64 default above instead of the MSYS2 just
  # checked.
  #
  # This file decides how the sub-project's configure is invoked on Windows
  # (therock_msys2_configure_srcdir below), so it belongs in the sub-project's
  # fingerprint, and an edit to it has to re-run the sub-project's build.  The
  # fingerprint is computed when the super-project configures, so it alone
  # would not do the latter, hence EXTRA_DEPENDS as well.
  set(${out_var}
    EXTERNAL_C_COMPILER "${_gcc}"
    EXTERNAL_CXX_COMPILER "${_gxx}"
    CMAKE_ARGS "-DTHEROCK_MSYS2_ROOT=${THEROCK_MSYS2_ROOT}"
    FPRINT_FILE_GLOBS "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
    EXTRA_DEPENDS "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
    PARENT_SCOPE)
endfunction()

# Paths passed to the wrapped command should stay in native Windows "C:/foo/bar"
# form (forward slashes), which is what CMake path variables already hold.
# MSYS2 bash accepts that form for both cd and executing a script, and the
# native ucrt64 gcc understands only the native form: an MSYS2 "/c/foo" mount
# path passed to gcc as an -I or #include target fails with "No such file or
# directory".  So do NOT convert paths to "/c/..." form before handing them to
# configure/make.
#
# One exception, handled by therock_msys2_configure_srcdir below: the path used
# to *invoke* configure (its srcdir) must be relative, not "C:/...".  This is a
# known autoconf >= 2.70 regression (unlikely to be fixed upstream): the
# generated configure splits its auxiliary-file search dir list on ':'
# (PATH_SEPARATOR), so a "C:/..." srcdir gets torn into "C" and "/...", and
# configure fails with "cannot find required auxiliary files: ltmain.sh ...".  A
# relative srcdir has no colon and sidesteps it, and unlike the common "pass a
# /c/... srcdir" workaround it does not reintroduce the gcc mount-path problem
# above (a relative path is fine for both).  It works for packages whose
# autoconf predates the split too.

# therock_msys2_configure_srcdir(<binary_dir> <source_dir> <out_var>)
#
# Compute a colon-free, relative path from <binary_dir> to
# <source_dir>/configure, for use as the configure invocation when the wrapped
# command runs with <binary_dir> as its cwd.  See the note above for why a
# relative path is required.
#
# When <source_dir> is on another drive than <binary_dir>, the path goes through
# a directory junction named therock-srcdir, created in <binary_dir>.
function(therock_msys2_configure_srcdir binary_dir source_dir out_var)
  file(RELATIVE_PATH _rel "${binary_dir}" "${source_dir}/configure")
  if(_rel MATCHES "^([A-Za-z]):")
    # A relative path cannot cross drives.  A sub-project whose source is
    # fetched under its own binary dir never needs one, but rocgdb's source is
    # the checkout's, and TheRock's Windows CI keeps the checkout and the build
    # tree on different drives.  A junction needs no privileges and works
    # between local volumes.  Deleting the build tree in Explorer, and with
    # cmake -E rm -rf, rmdir /s, PowerShell's Remove-Item -Recurse,
    # shutil.rmtree or Cygwin's rm -rf, all remove a junction without following
    # it, so wiping the build tree leaves the source alone.  The junction is
    # made afresh each time, in case the source dir has changed.
    set(_link "${binary_dir}/therock-srcdir")
    file(MAKE_DIRECTORY "${binary_dir}")
    file(TO_NATIVE_PATH "${_link}" _native_link)
    file(TO_NATIVE_PATH "${source_dir}" _native_source)
    if(IS_DIRECTORY "${_link}")
      # Without /s, rmdir removes the junction itself, never what it points to.
      execute_process(
        COMMAND cmd /c rmdir "${_native_link}"
        RESULT_VARIABLE _result
        ERROR_VARIABLE _error)
      if(NOT _result EQUAL 0)
        message(FATAL_ERROR
          "therock_msys2_configure_srcdir: cannot remove '${_link}': ${_error}")
      endif()
    endif()
    execute_process(
      COMMAND cmd /c mklink /J "${_native_link}" "${_native_source}"
      RESULT_VARIABLE _result
      OUTPUT_QUIET
      ERROR_VARIABLE _error)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "therock_msys2_configure_srcdir: cannot create a junction from "
        "'${_link}' to '${source_dir}', which is on another drive: ${_error}")
    endif()
    set(_rel "therock-srcdir/configure")
  endif()
  set("${out_var}" "${_rel}" PARENT_SCOPE)
endfunction()
