# ##############################################################################
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ##############################################################################
#
# Standalone test project for the rocm_ui C sources.
#
# Build and run:
#   cd runfile-installer/tests
#   cmake -B build -DCMAKE_BUILD_TYPE=Debug .
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# This project is intentionally separate from build-installer/CMakeLists.txt.
# It does not require INSTALLER_DIRECTORY, PACKAGE_NAME, or makeself — it is
# purely a developer testing tool.
# ##############################################################################

cmake_minimum_required(VERSION 3.16)
project(rocm_ui_tests C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

enable_testing()

# ── cmocka ────────────────────────────────────────────────────────────────────
# Use find_library rather than find_package so this works even on distros that
# ship cmocka without a CMake config file (only pkg-config or a bare .so).
find_library(CMOCKA_LIBRARY NAMES cmocka)
find_path(CMOCKA_INCLUDE_DIR NAMES cmocka.h)

if(NOT CMOCKA_LIBRARY OR NOT CMOCKA_INCLUDE_DIR)
    message(WARNING
        "cmocka not found — unit tests will not be built.\n"
        "Install with:\n"
        "  Ubuntu/Debian : sudo apt install libcmocka-dev\n"
        "  RHEL/Fedora   : sudo dnf install libcmocka-devel\n"
        "  SLES          : sudo zypper install libcmocka-devel")
    return()
endif()

message(STATUS "cmocka library : ${CMOCKA_LIBRARY}")
message(STATUS "cmocka include : ${CMOCKA_INCLUDE_DIR}")

# ── ncurses ───────────────────────────────────────────────────────────────────
# Dynamic linking is fine here — tests don't need the static portability that
# the production binary requires to run across distros.
find_package(Curses REQUIRED)

find_library(MENU_LIB NAMES menu)
find_library(FORM_LIB NAMES form)

if(NOT MENU_LIB OR NOT FORM_LIB)
    message(FATAL_ERROR
        "ncurses menu/form libraries not found.\n"
        "Install with:\n"
        "  Ubuntu/Debian : sudo apt install libncurses-dev\n"
        "  RHEL/Fedora   : sudo dnf install ncurses-devel\n"
        "  SLES          : sudo zypper install ncurses-devel")
endif()

message(STATUS "menu library   : ${MENU_LIB}")
message(STATUS "form library   : ${FORM_LIB}")

# ── Paths ─────────────────────────────────────────────────────────────────────
set(UI_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../UI/src")

# ── Production object library ─────────────────────────────────────────────────
# Compiles the UI modules exercised by the current test suites.
#
# Excluded:
#   rocm_ui.c      — defines main(); excluded to avoid duplicate main symbols.
#   driver_menu.c  — add when driver menu tests land.
#   pre_menu.c     — add when pre-install menu tests land.
#   post_menu.c    — add when post-install menu tests land.
#
# Compiled with the same C standard as production. -Werror is intentionally
# omitted so warnings in the production code do not gate developer test runs.
add_library(rocm_ui_testable OBJECT
    ${UI_SRC}/utils.c
    ${UI_SRC}/menu_data.c
    ${UI_SRC}/help_menu.c
    ${UI_SRC}/rocm_menu.c
)

target_include_directories(rocm_ui_testable PUBLIC
    ${UI_SRC}
    ${CURSES_INCLUDE_DIRS}
)

target_compile_options(rocm_ui_testable PRIVATE -Wall -Wextra)

# _DEFAULT_SOURCE exposes POSIX/GNU extensions (popen, strdup, etc.) that the
# production code uses.  With -std=c11 these are hidden without this macro.
target_compile_definitions(rocm_ui_testable PRIVATE _DEFAULT_SOURCE)

# ── Helper: register a test executable with CTest ─────────────────────────────
# Every test executable gets:
#   - the production object files (via TARGET_OBJECTS generator expression)
#   - test_stubs.c (NULL definitions for the globals owned by rocm_ui.c)
#   - cmocka, ncurses, menu, form, libm
function(add_rocm_test name source)
    add_executable(${name}
        ${source}
        ${CMAKE_CURRENT_SOURCE_DIR}/c/test_stubs.c
        $<TARGET_OBJECTS:rocm_ui_testable>
    )

    target_include_directories(${name} PRIVATE
        ${UI_SRC}
        ${CURSES_INCLUDE_DIRS}
        ${CMOCKA_INCLUDE_DIR}
    )

    target_link_libraries(${name} PRIVATE
        ${CMOCKA_LIBRARY}
        ${CURSES_LIBRARIES}
        ${MENU_LIB}
        ${FORM_LIB}
        m   # libmath — needed for ceil() in calculate_text_height()
    )

    # -Wno-unused-parameter: cmocka test functions require (void **state) but
    # simple tests that don't use setup/teardown state never read it.
    # Suppressing only for test executables, not for the production object library.
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wno-unused-parameter)

    # Match the object library: expose POSIX/GNU extensions (mkstemp, write,
    # fdopen, ssize_t, etc.) that are hidden under strict -std=c11 without this.
    target_compile_definitions(${name} PRIVATE _DEFAULT_SOURCE)

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ── Test targets ──────────────────────────────────────────────────────────────
# Pure-logic functions — no production code changes required to build these.
add_rocm_test(test_utils          c/test_utils.c)
add_rocm_test(test_gfx_extraction c/test_gfx_extraction.c)
add_rocm_test(test_file_io        c/test_file_io.c)
