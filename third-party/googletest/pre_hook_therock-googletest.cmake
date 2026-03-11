# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# We want to be able to use the static build gtest lib in shared libraries.
message(STATUS "Enabling PIC build")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
