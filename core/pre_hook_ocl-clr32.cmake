# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for the 32-bit OpenCL CLR build.

# LLVMConfig.cmake guards its include(LLVMExports.cmake) behind
# "NOT CMAKE_CROSSCOMPILING". The 32-bit toolchain sets
# CMAKE_CROSSCOMPILING=TRUE (x64 host → x86 target), which causes
# find_package(LLVM/Clang/LLD) to silently skip importing targets.
# Override so the installed LLVM packages load correctly.
set(CMAKE_CROSSCOMPILING FALSE)
