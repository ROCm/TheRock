# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RBT's CMakeLists.txt only sets CMAKE_MODULE_PATH if it is not already
# defined. When building as a TheRock subproject, the variable is pre-set by
# the toolchain/init infrastructure, so RBT's own cmake_modules/ directory is
# never added and `include(utils)` fails. Append it here so both TheRock's
# modules and RBT's modules are available.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
