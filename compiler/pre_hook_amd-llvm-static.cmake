# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-hook for the `amd-llvm-static` subproject: a second, build-only build of
# LLVM/Clang/LLD configured for STATIC linking so that amd-comgr can embed a
# single, isolated copy of LLVM.
#
# Why this exists
# ---------------
# The shipped `amd-llvm` build uses LLVM_LINK_LLVM_DYLIB=ON (see
# pre_hook_amd-llvm.cmake). In that configuration the exported clang/lld static
# library targets carry an INTERFACE dependency on libLLVM.so. comgr is built
# with COMGR_STATIC_LLVM=ON, which also adds the LLVM/Clang component *static*
# archives to comgr's link line. Linking both pulls TWO copies of LLVM into
# libamd_comgr.so -- the static archives AND libLLVM.so -- whose independent
# global state (cl::opt registry, ManagedStatic, the Target/Pass registries)
# corrupts comgr's in-process compile/codegen/lld pipeline at runtime.
#
# There is no link-time fix on comgr's side: a clean static comgr requires
# clang/lld libraries that do NOT carry the libLLVM.so interface, which only
# exist when LLVM is built with LLVM_LINK_LLVM_DYLIB=OFF. Rather than flip the
# whole shipped toolchain to static (which would bloat every clang/lld/flang
# binary), we build a dedicated static LLVM used ONLY as comgr's build-time link
# dependency. It is not shipped.
#
# This hook reuses the main amd-llvm configuration and overrides only what is
# needed to (a) link statically and (b) trim the build to the projects comgr
# actually links.

# Start from the shipped amd-llvm configuration so the two builds stay in sync
# on targets, device-libs, spirv-translator, version stamping, RPATH, etc.
include("${CMAKE_CURRENT_LIST_DIR}/pre_hook_amd-llvm.cmake")

# --- Static link override (the whole point of this subproject) ---------------
# Build the component static archives only; do not build or link libLLVM.so.
# With LLVM_LINK_LLVM_DYLIB=OFF the clang/lld library targets link the LLVM
# component archives and their INTERFACE_LINK_LIBRARIES no longer references the
# `LLVM` dylib, so comgr ends up with exactly one copy of LLVM.
set(LLVM_BUILD_LLVM_DYLIB OFF)
set(LLVM_LINK_LLVM_DYLIB OFF)

# --- Scope trimming ----------------------------------------------------------
# comgr links clang, lld and the LLVM/SPIRV component libraries. It does not
# need flang or clang-tools-extra, and it does not link the device/offload
# runtimes (those ship from the main amd-llvm build). Trimming keeps the cost of
# the second build down.
#
# NOTE: keep this list in sync with what comgr actually links (src/CMakeLists.txt
# CLANG_LIBS / LLD_LIBS / LLVM component list). VALIDATE against a real build.
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "Enable LLVM projects" FORCE)

# comgr links host libraries only; the device/offload runtimes are not needed in
# this build. Dropping the runtimes also lets us build against the system C++
# library instead of requiring the libcxx runtime to be built here.
set(LLVM_ENABLE_RUNTIMES "" CACHE STRING "Enabled runtimes" FORCE)
set(LLVM_ENABLE_LIBCXX OFF)
