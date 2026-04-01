# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(cov-backward-compat LANGUAGES CXX)

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
find_package(hip REQUIRED CONFIG)
find_package(hiprtc REQUIRED CONFIG)

add_executable(cov_backward_compat cov_backward_compat_hiprtc.cpp)
target_link_libraries(cov_backward_compat PRIVATE hip::host hiprtc::hiprtc)

include(GNUInstallDirs)
install(TARGETS cov_backward_compat RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/cov-tests)
