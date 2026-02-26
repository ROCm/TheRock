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

cmake_minimum_required(VERSION 3.6)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION_FILE_CONTENT)
string(REPLACE "\n" ";" VERSION_STRING_LIST ${VERSION_FILE_CONTENT})
list(GET VERSION_STRING_LIST 0 VERSION_STRING)
list(GET VERSION_STRING_LIST 1 ROCM_VERSION_STRING)

project(rocm_runfile_installer VERSION ${VERSION_STRING} LANGUAGES C)

# Dependency check
find_program(MAKESELF makeself)


message(STATUS "+++++++++++++++++++++++++++++++++++++")
message(STATUS "ROCm Runfile Installer")
message(STATUS "+++++++++++++++++++++++++++++++++++++")

message(STATUS "VERSION_STRING      = " ${VERSION_STRING})
message(STATUS "ROCM_VERSION_STRING = " ${ROCM_VERSION_STRING})
message(STATUS "INSTALLER_NAME      = " ${INSTALLER_NAME})
message(STATUS "AMDGPU_DKMS         = " ${AMDGPU_DKMS})
message(STATUS "MAKESELF_VERSION    = " ${MAKESELF_VERSION})
message(STATUS "MAKESELF_OPTIONS    = " ${MAKESELF_OPTIONS})
message(STATUS "INCLUDE_DIRECTORIES            = " ${INCLUDE_DIRECTORIES})
message(STATUS "CMAKE_SOURCE_DIR               = " ${CMAKE_SOURCE_DIR})
message(STATUS "CMAKE_BINARY_DIR               = " ${CMAKE_BINARY_DIR})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR       = " ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "CMAKE_CURRENT_BINARY_DIR       = " ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS "CMAKE_CURRENT_LIST_DIR         = " ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "CMAKE_BINARY_BINDIR            = " ${CMAKE_BINARY_BINDIR})
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY = " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
message(STATUS "CMAKE_BUILD_TYPE               = " ${CMAKE_BUILD_TYPE})

#enable_testing()

# Check for the Distro
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    execute_process (
        COMMAND bash -c "awk -F= '/^ID=/{print $2}' /etc/os-release |tr -d '\n' | tr -d '\"'"
        OUTPUT_VARIABLE DISTRO
    )
    message(STATUS "Linux distro: ${DISTRO}")

    execute_process (
        COMMAND bash -c "awk -F= '/^VERSION_ID=/{print $2}' /etc/os-release |tr -d '\n' | tr -d '\"'"
        OUTPUT_VARIABLE DISTRO_VERSION
    )
    message(STATUS "Linux distro version: ${DISTRO_VERSION}")
else()
    message(FATAL_ERROR "No OS detected!")
endif()

# Set the script directory for the distro
# Note: DISTRO_BUILD_STRING is no longer passed to UI (reads VERSION file at runtime)
if(${DISTRO} STREQUAL "ubuntu")
    message(STATUS "Building for Ubuntu")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "Ubuntu ${DISTRO_VERSION}")  # No longer used by UI
elseif(${DISTRO} STREQUAL "debian")
    message(STATUS "Building for Debian")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "Debian ${DISTRO_VERSION}")  # No longer used by UI
elseif(${DISTRO} STREQUAL "rhel")
    message(STATUS "Building for RHEL")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "Red Hat Enterprise Linux ${DISTRO_VERSION}")  # No longer used by UI
elseif(${DISTRO} STREQUAL "ol")
    message(STATUS "Building for Oracle Linux")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "Oracle Linux ${DISTRO_VERSION}")  # No longer used by UI
elseif(${DISTRO} STREQUAL "rocky")
    message(STATUS "Building for Rocky Linux")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "Rocky Linux ${DISTRO_VERSION}")  # No longer used by UI
elseif(${DISTRO} STREQUAL "sles" OR ${DISTRO} STREQUAL "sled")
    message(STATUS "Building for Suse")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "SUSE Linux Enterprise Server ${DISTRO_VERSION}")  # No longer used by UI
elseif(${DISTRO} STREQUAL "amzn")
    message(STATUS "Building for Amazon")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "Amazon Linux ${DISTRO_VERSION}")  # No longer used by UI
elseif(${DISTRO} STREQUAL "almalinux")
    message(STATUS "Building for Almalinux")
    set(INSTALLER_DIRECTORY ${CMAKE_SOURCE_DIR}/../rocm-installer)
    # set(DISTRO_BUILD_STRING "Almalnux ${DISTRO_VERSION}")  # No longer used by UI
else()
    message(FATAL_ERROR "Unsupported Distro : ${DISTRO}")
endif()

# Set the Installer package name
set(PACKAGE_NAME ${INSTALLER_NAME})

message(STATUS "INSTALLER_DIRECTORY = " ${INSTALLER_DIRECTORY})
message(STATUS "PACKAGE_NAME        = " ${PACKAGE_NAME})

# Define the targets
add_custom_target(rocm_installer ALL
                  COMMAND echo "Copying version file"
                  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/VERSION ${INSTALLER_DIRECTORY}/
                  COMMAND echo "Building rocm_installer")
                  #COMMAND ${CMAKE_SOURCE_DIR}/build-installer.sh)
#COMMAND makeself ${MAKESELF_OPTIONS} --nox11 ${INSTALLER_DIRECTORY} ${PACKAGE_NAME}.run "ROCm Runfile Installer" ./install_init.sh)

# UI
add_subdirectory(../UI UI)

# Testing
#add_subdirectory(tests)
