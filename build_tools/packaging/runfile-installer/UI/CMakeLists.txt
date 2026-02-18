# ##############################################################################
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
#
# ##############################################################################

message(STATUS "+++++++++++++++++++++++++++++++++++++++++++")
message(STATUS "Creating for ROCm Runfile Installer UI")
message(STATUS "+++++++++++++++++++++++++++++++++++++++++++")

message(STATUS "UI: CMAKE_SOURCE_DIR               = " ${CMAKE_SOURCE_DIR})
message(STATUS "UI: CMAKE_BINARY_DIR               = " ${CMAKE_BINARY_DIR})
message(STATUS "UI: CMAKE_CURRENT_SOURCE_DIR       = " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "UI: CMAKE_CURRENT_BINARY_DIR       = " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "UI: CMAKE_CURRENT_LIST_DIR         = " ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "VERSION_STRING                     = " ${VERSION_STRING})
message(STATUS "DISTRO_BUILD_STRING                = " ${DISTRO_BUILD_STRING})
message(STATUS "AMDGPU_DKMS                        = " ${AMDGPU_DKMS})

message(STATUS "Installer_directory = " ${INSTALLER_DIRECTORY})
message(STATUS "UI binary location  = " ${CMAKE_BINARY_DIR})

# Find ncurses related items - STATIC versions for portability
find_package(Curses REQUIRED)

# Find static libraries - REQUIRED for cross-distro portability
find_library(NCURSES_STATIC_LIB NAMES libncurses.a PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/lib
    NO_DEFAULT_PATH)

find_library(MENU_STATIC_LIB NAMES libmenu.a PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/lib
    NO_DEFAULT_PATH)

find_library(FORM_STATIC_LIB NAMES libform.a PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/lib
    NO_DEFAULT_PATH)

find_library(TINFO_STATIC_LIB NAMES libtinfo.a PATHS
    /usr/lib/x86_64-linux-gnu
    /usr/lib64
    /usr/lib
    NO_DEFAULT_PATH)

# Verify static libraries are found - ERROR if not (don't fall back to dynamic)
if(NOT NCURSES_STATIC_LIB)
    message(FATAL_ERROR "Static ncurses library (libncurses.a) not found! Install ncurses-static package.")
endif()

if(NOT MENU_STATIC_LIB)
    message(FATAL_ERROR "Static menu library (libmenu.a) not found! Install ncurses-static package.")
endif()

if(NOT FORM_STATIC_LIB)
    message(FATAL_ERROR "Static form library (libform.a) not found! Install ncurses-static package.")
endif()

message(STATUS "NCURSES_STATIC_LIB = ${NCURSES_STATIC_LIB}")
message(STATUS "MENU_STATIC_LIB    = ${MENU_STATIC_LIB}")
message(STATUS "FORM_STATIC_LIB    = ${FORM_STATIC_LIB}")
message(STATUS "TINFO_STATIC_LIB   = ${TINFO_STATIC_LIB}")

include_directories(${CURSES_INCLUDE_DIR} ${CURSES_INCLUDE_DIR}/ncurses)

file(GLOB SOURCEFILES "src/*.c")

add_executable(rocm_ui ${SOURCEFILES})

# Version info now loaded at runtime from VERSION file - no compile-time definitions needed

# UI target settings
target_compile_options(rocm_ui PRIVATE -Wall -Wextra -Werror)

# Link static libraries with proper dependency order
# Order matters: menu/form depend on ncurses, ncurses may depend on tinfo
# Use -Wl,-Bstatic to force static linking of ncurses libraries
target_link_libraries(rocm_ui
    -Wl,-Bstatic
    ${MENU_STATIC_LIB}
    ${FORM_STATIC_LIB}
    ${NCURSES_STATIC_LIB}
)

# Add tinfo if found (required by ncurses on some systems)
if(TINFO_STATIC_LIB)
    target_link_libraries(rocm_ui ${TINFO_STATIC_LIB})
endif()

# Allow dynamic linking for system libraries (dl, pthread, etc.)
target_link_libraries(rocm_ui -Wl,-Bdynamic)

# Add system libraries required by static ncurses
# dl library is needed for dlopen/dlsym/dlclose (used by ncurses for GPM)
target_link_libraries(rocm_ui dl)

# Force static linking for libgcc
set_target_properties(rocm_ui PROPERTIES LINK_FLAGS "-static-libgcc")

add_dependencies(rocm_installer rocm_ui)

add_custom_command(TARGET rocm_ui
                  COMMAND echo "Copying UI to installer directory"
                  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/rocm_ui ${INSTALLER_DIRECTORY}/)
