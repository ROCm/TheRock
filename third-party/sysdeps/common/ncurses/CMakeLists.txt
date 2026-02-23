if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-ncurses-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://invisible-island.net/archives/ncurses/ncurses-6.5.tar.gz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/ncurses-6.5.tar.gz"
      URL_HASH "SHA512=fc5a13409d2a530a1325776dcce3a99127ddc2c03999cfeb0065d0eee2d68456274fb1c7b3cc99c1937bc657d0e7fca97016e147f93c7821b5a4a6837db821e8"
      TOUCH "${_download_stamp}"
      # We need DOWNLOAD_EXTRACT_TIMESTAMP set to TRUE to preserve the timestamp
      # of the extracted sources, otherwise the build system will notice
      # changes and will try to re-run autotools, causing issues.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    therock_cmake_subproject_declare(therock-ncurses
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

    set(NCURSES_HEADERS
      "ncurses.h"
      "term.h"
      "curses.h"
      "cursesp.h"
      "panel.h"
      "cursslk.h"
      "etip.h"
      "cursesf.h"
      "cursesapp.h"
      "eti.h"
      "cursesm.h"
      "cursesw.h"
      "ncurses_dll.h"
      "menu.h"
      "term_entry.h"
      "form.h"
      "unctrl.h"
      "termcap.h"
    )

    foreach(ncurses_header ${NCURSES_HEADERS})
      add_test(
        NAME ncurses_header_test:${ncurses_header}
        COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/${ncurses_header}"
      )
    endforeach()

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(NCURSES_BUILD VERSION "6.5.0")

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  #
  # OPTIONAL: Patching of the sources should happen here.
  #
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}"
      # Escaping: Double $ to satisfy CMake, then double $ to satisfy configure,
      # then escaped single quotes to make it to the linker command line.
      "LDFLAGS=${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath=\\'\\$$\\$$ORIGIN:\\$$\\$$ORIGIN/../lib\\' -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
      # Add -std=gnu17 so workaround GCC 15 build issues for now.
      # See https://github.com/conan-io/conan-center-index/issues/28083
      "CPPFLAGS=${EXTRA_CPPFLAGS} -std=gnu17"
      --
    "${SOURCE_DIR}/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --with-shared
      --without-ada
      --enable-widec
      --enable-pc-files
      # Without this, it will attempt to install pcfiles in the path reported
      # by pkgconfig (which is typically the system root, which is not what
      # we want).
      --with-pkg-config-libdir="${CMAKE_INSTALL_PREFIX}/lib/pkgconfig"
      # Prevent build-time LDFLAGS (rpath, version-script) from leaking into
      # the .pc files' Libs: line.
      --disable-pkg-ldflags
      # Enable overwriting of header files in the default installation prefix.
      # This makes it so we install the ncurses headers in the include/
      # directory as opposed to a distinct subdirectory.
      --enable-overwrite
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
      --with-fallbacks="xterm,xterm-256color,vt100,linux,screen,screen-256color,tmux,tmux-256color,ansi"
      # Since we have the fallback above, set the terminfo lookup paths to the
      # standard installation prefixes. We hope the system will provide the
      # additional required terminfo files if the fallbacks are not enough.
      --with-terminfo-dirs="/usr/share/terminfo:/usr/lib/terminfo:/etc/terminfo:/lib/terminfo"
      # Fix #3223: Override autoconf's pid_t detection which incorrectly
      # defines pid_t as int on modern systems with GCC 14+. This conflicts
      # with glibc's typedef, breaking wchar_t detection and causing
      # --enable-widec to silently fail.
      ac_cv_type_pid_t=yes
  COMMAND
    make -j "${PAR_JOBS}" V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ncurses-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ncurses-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ncurses-config.cmake" DESTINATION lib/cmake/ncurses)
