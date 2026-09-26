# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for `amd-llvm-static`: a build-only build of LLVM/Clang/LLD that
# amd-comgr links statically.
#
# The shipped amd-llvm build sets LLVM_LINK_LLVM_DYLIB=ON, which makes its
# exported clang/lld library targets carry an INTERFACE dependency on
# libLLVM.so. comgr is built with COMGR_STATIC_LLVM=ON, which additionally puts
# the LLVM component archives on comgr's link line, so linking those clang/lld
# targets would embed two copies of LLVM in libamd_comgr.so. The copies have
# independent global state (cl::opt registry, ManagedStatic, the target and
# pass registries), which breaks comgr's in-process compiler at runtime.
#
# Clang/lld libraries without the libLLVM.so interface only exist in a build
# configured with LLVM_LINK_LLVM_DYLIB=OFF, hence this second build. Flipping
# the shipped toolchain to static instead would bloat every clang/lld/flang
# binary.

# Inherit the shipped configuration so both builds stay in sync on targets,
# device-libs, spirv-translator, version stamping and RPATH.
include("${CMAKE_CURRENT_LIST_DIR}/pre_hook_amd-llvm.cmake")

# The point of this subproject: component archives only, no libLLVM.so.
set(LLVM_BUILD_LLVM_DYLIB OFF)
set(LLVM_LINK_LLVM_DYLIB OFF)

# comgr links clang, lld and the LLVM/SPIRV component libraries, all host-side.
# Drop flang, clang-tools-extra and the runtimes to keep this build cheap.
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "Enable LLVM projects" FORCE)
set(LLVM_ENABLE_RUNTIMES "" CACHE STRING "Enabled runtimes" FORCE)
set(LLVM_INCLUDE_RUNTIMES OFF)
