# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for the 32-bit comgr build. Tests are disabled; only the shared
# library (amd_comgr32.dll) is produced.

set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)

# LLVMConfig.cmake guards its include(LLVMExports.cmake) — which defines all
# LLVM imported targets — behind "NOT CMAKE_CROSSCOMPILING". The 32-bit
# toolchain sets CMAKE_CROSSCOMPILING=TRUE (x64 host → x86 target), so the
# exports are skipped and LLDConfig/ClangConfig fail with missing targets.
# Override for the remainder of this subproject: the 32-bit compiler and
# linker are already locked in by the toolchain, and the LLVM/Clang/LLD
# packages here come from the same 32-bit build, so loading their exports
# is safe.
set(CMAKE_CROSSCOMPILING FALSE)

if(WIN32)
  set(CMAKE_INSTALL_RPATH "")
else()
  set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/llvm/lib;$ORIGIN/rocm_sysdeps/lib")
endif()
