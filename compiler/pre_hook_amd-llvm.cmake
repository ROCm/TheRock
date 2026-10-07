# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TheRock-only overrides for amd-llvm.
#
# LLVM projects, runtimes, targets, distributions, tool allow-lists, external
# projects (device-libs / SPIR-V / sqtt), install RPATH, and most OpenMP/Flang
# options come from clang/cmake/caches/TheRock.cmake (via -C). Do not FORCE
# those here or they will override the cache.
#
# Note that in LLVM "BUILD_SHARED_LIBS" enables an unsupported development mode.
# The flag you want for a shared library build is LLVM_BUILD_LLVM_DYLIB (set by
# the cache).
set(BUILD_SHARED_LIBS OFF)

if(NOT WIN32)
  # There is an issue with finding the zstd config built by TheRock when zstd
  # is searched for in the llvm config. LLVM has a FindZSTD.cmake that is
  # found in module mode, which ultimately fails to locate the library.
  # Prefer CONFIG mode for runtime ExternalProjects.
  set(RUNTIMES_CMAKE_ARGS "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON")

  # Use DWARF4 for sanitizer builds. dwz (the DWARF optimization tool used in
  # Debian/Ubuntu packaging) doesn't fully support DWARF5 - it fails with
  # "Unknown debugging section .debug_str_offsets" even in version 0.16
  # (Ubuntu 26.04). This is an upstream dwz limitation, not something we
  # can fix by updating distro packages. Revisit if dwz gains DWARF5 support.
  if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN" OR THEROCK_SANITIZER STREQUAL "TSAN")
    string(APPEND RUNTIMES_CMAKE_ARGS ";-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS} -gdwarf-4;-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -gdwarf-4")
  endif()

  # Hosts whose C long double is already IEEE-754 binary128 (ppc64le,
  # riscv64) do not need libquadmath: flang-rt folds the F128 entry points
  # into libflang_rt.runtime instead of building libflang_rt.quadmath.
  #
  # This must be a typed cache entry, not a normal variable.
  # flang/cmake/modules/FlangCommon.cmake declares the variable with
  # set(... CACHE STRING) and LLVM sets cmake_minimum_required(3.20), so
  # CMP0126 is OLD there and that call would discard a normal variable of
  # the same name, leaving the flang driver and the flang-rt runtimes build
  # with different values. An existing typed cache entry makes the
  # FlangCommon.cmake declaration a no-op, so this selection is
  # authoritative for both.
  #
  # Kept here (not only in ROCm.cmake) so FORCE wins before flang configures,
  # including for riscv64 which the cache file may not yet special-case.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|riscv64")
    set(FLANG_RUNTIME_F128_MATH_LIB "" CACHE STRING "Library implementing REAL(16) math for flang-rt" FORCE)
  else()
    set(FLANG_RUNTIME_F128_MATH_LIB "libquadmath" CACHE STRING "Library implementing REAL(16) math for flang-rt" FORCE)
  endif()

  # TODO: Enable when HWLOC dependency is figured out.
  # set(LIBOMP_USE_HWLOC ON)
endif()

# TODO2: This mechanism has races in certain situations, failing to create a
# symlink. Revisit once devicemanager code is made more robust.
# TODO: Arrange for the devicelibs to be installed to the clange resource dir
# by default. This corresponds to the layout for ROCM>=7. However, not all
# code (specifically the AMDDeviceLibs.cmake file) has adapted to the new
# location, so we have to also make them available at amdgcn. There are cache
# options to manage this transition but they require knowing the clange resource
# dir. In order to avoid drift, we just fixate that too. This can all be
# removed in a future version.
# set(CLANG_RESOURCE_DIR "../lib/clang/${LLVM_VERSION_MAJOR}" CACHE STRING "Resource dir" FORCE)
# set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW "lib/clang/${LLVM_VERSION_MAJOR}/amdgcn" CACHE STRING "New devicelibs loc" FORCE)
# set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD "amdgcn" CACHE STRING "Old devicelibs loc" FORCE)

# Keep CTest/BUILD_TESTING aligned with TheRock's LLVM test toggle. The cache
# uses LLVM_INCLUDE_TESTS (passed from compiler/CMakeLists.txt) for tool/test
# policy; this covers subprojects that still key off BUILD_TESTING.
if(THEROCK_BUILD_LLVM_TESTS)
  set(BUILD_TESTING ON CACHE BOOL "Enable building LLVM tests" FORCE)
else()
  set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)
endif()
