# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-ncurses-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://invisible-island.net/archives/ncurses/ncurses-6.6.tar.gz
      #
      # On a version bump, re-read the mingw references cited in the Windows
      # configure branch below, and check whether the recipe needs adjusting,
      # and whether patches/third-party/ncurses is still needed.
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/ncurses-6.6.tar.gz"
      URL_HASH "SHA512=02647baae53abc844fbadee5b0a2187ad073125c4e8950df6d1c4feb781cb74ba64fb838cedfee2c246c39932187f6775b1df124f18b99a4233f0d98c72191de"
      PATCH_COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/patch_third_party_source.py ${CMAKE_SOURCE_DIR}/patches/third-party/ncurses
      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # An autotools sysdep is built by the MSYS2 ucrt64 gcc on Windows, not by
    # the ambient MSVC, so name that toolchain: without it CMAKE_C_COMPILER is
    # cl.exe, and the build would depend on the login shell's PATH to supply a
    # usable compiler.
    set(_sysdep_toolchain)
    if(WIN32)
      set(_sysdep_toolchain host-gnu)
    endif()
    therock_autotools_subproject_args(therock-ncurses _autotools_args)

    therock_cmake_subproject_declare(therock-ncurses
      HARDENING_DEFERRED "ncurses (autotools)"
      ${_autotools_args}
      COMPILER_TOOLCHAIN ${_sysdep_toolchain}
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR .
      FPRINT_SOURCE_DIR "${_source_dir}"
      FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-ncurses ncurses lib/rocm_sysdeps/lib/cmake/ncurses)
    therock_cmake_subproject_activate(therock-ncurses)

    # On Windows the libraries are built static (.a), so there are no shared
    # objects to validate.
    if(NOT WIN32)
      therock_test_validate_shared_lib(
        PATH build/dist/lib/rocm_sysdeps/lib
        LIB_NAMES libmenuw.so
      )

      therock_test_validate_shared_lib(
        PATH build/dist/lib/rocm_sysdeps/lib
        LIB_NAMES libncursesw.so
      )

      therock_test_validate_shared_lib(
        PATH build/dist/lib/rocm_sysdeps/lib
        LIB_NAMES libpanelw.so
      )

      therock_test_validate_shared_lib(
        PATH build/dist/lib/rocm_sysdeps/lib
        LIB_NAMES libformw.so
      )
    endif()

    set(NCURSES_HEADERS
      "ncurses.h"
      "term.h"
      "curses.h"
      "panel.h"
      "eti.h"
      "ncurses_dll.h"
      "menu.h"
      "term_entry.h"
      "form.h"
      "unctrl.h"
      "termcap.h"
    )
    # The C++ binding's headers.  Windows builds no C++ binding (see its
    # configure arguments below).
    if(NOT WIN32)
      list(APPEND NCURSES_HEADERS
        "cursesp.h"
        "cursslk.h"
        "etip.h"
        "cursesf.h"
        "cursesapp.h"
        "cursesm.h"
        "cursesw.h"
      )
    endif()

    # Checked through cmake instead of ls, which a ctest run from cmd.exe or
    # PowerShell on Windows does not have.  md5sum fails on a missing file.
    foreach(ncurses_header ${NCURSES_HEADERS})
      add_test(
        NAME ncurses_header_test:${ncurses_header}
        COMMAND "${CMAKE_COMMAND}" -E md5sum "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/${ncurses_header}"
      )
    endforeach()

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(NCURSES_BUILD VERSION "6.6.0")

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

include("${THEROCK_SOURCE_DIR}/cmake/therock_autotools.cmake")

set(configure_env)
set(pre_commands)
set(mid_commands)
set(post_commands)

# Arguments both platforms configure with: widec for wide characters,
# pc-files under the install prefix, and --enable-overwrite so that headers
# install into include/ directly.  The branch below adds only what differs.
set(configure_args
  "--prefix=${CMAKE_INSTALL_PREFIX}"
  "--libdir=${CMAKE_INSTALL_PREFIX}/lib"
  --without-ada
  --enable-widec
  --enable-pc-files
  # Without this, it will attempt to install pcfiles in the path reported
  # by pkgconfig (which is typically the system root, which is not what
  # we want).
  "--with-pkg-config-libdir=${CMAKE_INSTALL_PREFIX}/lib/pkgconfig"
  # Direct terminfo database installation into the stage tree so
  # make install does not attempt to write to /usr/share/terminfo.
  "--with-default-terminfo-dir=${CMAKE_INSTALL_PREFIX}/share/terminfo"
  # Prevent build-time LDFLAGS (rpath, version-script) from leaking into
  # the .pc files' Libs: line.
  --disable-pkg-ldflags
  # Enable overwriting of header files in the default installation prefix.
  # This makes it so we install the ncurses headers in the include/
  # directory as opposed to a distinct subdirectory.
  --enable-overwrite)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # ncurses builds static here, providing the libraries (libncursesw.a etc.)
  # that rocgdb's TUI links against.
  #
  # Native Windows ncurses needs the runtime terminal-driver abstraction
  # (--enable-term-driver, which requires --enable-sp-funcs) so that it drives
  # the Win32 console instead of assuming a POSIX tty.  This, plus
  # --disable-home-terminfo / --enable-database / --enable-interop and the two
  # workarounds below, follows ncurses' own README.MinGW and the MSYS2
  # mingw-w64-ncurses PKGBUILD, found at:
  #
  #   https://github.com/mirror/ncurses/blob/master/README.MinGW
  #   https://github.com/msys2/MINGW-packages/blob/master/mingw-w64-ncurses/PKGBUILD
  #
  # Note that both track master instead of a tagged release, so what they say is
  # guidance for whatever ncurses is current, not necessarily for the version we
  # fetched above.  Re-read them when the version we pull is updated.
  #
  #   - -D__USE_MINGW_ACCESS works around a Windows access() quirk.  It is
  #     passed in CPPFLAGS, not CFLAGS: setting CFLAGS at all would drop
  #     autoconf's default -g -O2.
  #   - cf_cv_func_nanosleep=no: mingw's nanosleep is unreliable, so force
  #     ncurses to use its own timing.
  #
  # There is no C++ binding (--without-cxx-binding).  Nothing in TheRock uses
  # it, rocgdb's TUI being written against the C API, and built by mingw g++ it
  # would be a C++ library that none of the MSVC-built components could link
  # either, the two C++ ABIs being incompatible.  README.MinGW leaves out C++
  # altogether, with --without-cxx.
  #
  # The terminfo fallbacks keep the TUI working for common terminals when the
  # terminfo database is not found -- important for a relocatable native install
  # with no /usr/share/terminfo.
  set(configure_env
    "CPPFLAGS=-D__USE_MINGW_ACCESS"
    cf_cv_func_nanosleep=no)
  list(APPEND configure_args
    --enable-static --disable-shared
    --without-cxx-binding
    --enable-sp-funcs
    --enable-term-driver
    --enable-interop
    --enable-database
    --disable-home-terminfo
    --with-fallbacks=ms-terminal,xterm,xterm-256color,vt100,ansi)

  list(APPEND post_commands
    COMMAND
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX})

else()

  # Apply patch source only on Linux
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND pre_commands   COMMAND
      bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}")

    list(APPEND mid_commands   COMMAND
      bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  set(configure_env
    "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
    # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
    # then escaped single quotes to make it to the linker command line.
    "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN:\\$$\\$$ORIGIN/../lib\\' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
    # Add -std=gnu17 so workaround GCC 15 build issues for now.
    # See https://github.com/conan-io/conan-center-index/issues/28083
    "CPPFLAGS=${EXTRA_CPPFLAGS} -std=gnu17")
  list(APPEND configure_args
    --with-shared
    # Private names for the shared form, menu and panel libraries.
    # patch_install.py gives them back their usual link and pkg-config names.
    --with-form-libname=rocm_sysdeps_form
    --with-menu-libname=rocm_sysdeps_menu
    --with-panel-libname=rocm_sysdeps_panel
    # Enable fallbacks. We are building our own ncurses library. Though we
    # install it at a known place in our TheRock installation tree, the tree
    # itself can be located anywhere (relocatable).
    #
    # The problem is that terminfo lookup logic in ncurses is rigid and
    # expects (a list of) absolute directories, and we can't provide that.
    #
    # For this reason, the fallback should act as a mechanism that allows
    # ROCgdb's TUI interface to work for the majority of the most common
    # terminal types even when the terminfo database cannot be found.
    --with-fallbacks=xterm,xterm-256color,vt100,linux,screen,screen-256color,tmux,tmux-256color,ansi
    # Since we have the fallback above, set the terminfo lookup paths to the
    # standard installation prefixes. We hope the system will provide the
    # additional required terminfo files if the fallbacks are not enough.
    --with-terminfo-dirs=/usr/share/terminfo:/usr/lib/terminfo:/etc/terminfo:/lib/terminfo
    # Fix #3223: Override autoconf's pid_t detection which incorrectly
    # defines pid_t as int on modern systems with GCC 14+. This conflicts
    # with glibc's typedef, breaking wchar_t detection and causing
    # --enable-widec to silently fail.
    ac_cv_type_pid_t=yes)

  list(APPEND post_commands
    COMMAND
      "${CMAKE_COMMAND}" -E env
        "PATCHELF=${PATCHELF}"
        "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
        "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX})
endif()

therock_autotools_shell(
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}"
  SOURCE_DIR "${SOURCE_DIR}"
  CONFIGURE_ENV ${configure_env}
  RUN_VAR run
  RUN_CONFIGURE_VAR run_configure
  CONFIGURE_ARGS_VAR host_args
  CONFIGURE_VAR configure)

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  ${pre_commands}
  COMMAND ${run_configure} ${configure} ${host_args} ${configure_args}
  ${mid_commands}
  COMMAND ${run} make -j "${PAR_JOBS}" V=1
  COMMAND ${run} make install
  ${post_commands}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ncurses-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ncurses-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ncurses-config.cmake" DESTINATION lib/cmake/ncurses)
