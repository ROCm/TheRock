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
