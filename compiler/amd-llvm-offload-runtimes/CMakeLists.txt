cmake_minimum_required(VERSION 3.20)
project(amd-llvm-runtimes C CXX ASM)

# Source directory for the LLVM monorepo
set(LLVM_MONOREPO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../amd-llvm")

#=============================================================================
# Find pre-built LLVM to get version info needed by runtimes
#=============================================================================
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

# PACKAGE_VERSION is needed by GetClangResourceDir.cmake
set(PACKAGE_VERSION ${LLVM_PACKAGE_VERSION})

# Disable tests - they depend on LLVM-internal targets like llvm_gtest
# In order to enable (if desired), we need to install into a non-distributed
# component that we can still depend on here.
set(LLVM_INCLUDE_TESTS OFF)

# Runtimes to build - set as CACHE FORCE to override the default empty value
# in runtimes/CMakeLists.txt. Set here to avoid semicolon escaping issues in CMAKE_ARGS.
set(LLVM_ENABLE_RUNTIMES "openmp;offload;flang-rt" CACHE STRING "" FORCE)

# Skip Fortran compiler test - flang-rt doesn't exist yet so the test would fail
# Also set compiler ID so CMake knows how to invoke flang properly
set(CMAKE_Fortran_COMPILER_WORKS YES CACHE BOOL "" FORCE)

# Misc policy settings.
set(OPENMP_ENABLE_LIBOMPTARGET ON)
set(LIBOMPTARGET_BUILD_DEVICE_FORTRT ON)
set(LIBOMPTARGET_ENABLE_DEBUG ON)
set(LIBOMPTARGET_NO_SANITIZER_AMDGPU ON)
set(LIBOMP_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../lib:\$ORIGIN/../../lib:\$ORIGIN/../../../lib")
# Setting "LIBOMP_COPY_EXPORTS" to `OFF` "aids parallel builds to not interfere
# with each other" as libomp and generated headers are copied into the original
# source otherwise. Defaults to `ON`.
set(LIBOMP_COPY_EXPORTS OFF)

set(LLVM_RUNTIME_TARGETS "default;amdgcn-amd-amdhsa")
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES "openmp")
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON)
set(FLANG_RUNTIME_F128_MATH_LIB "libquadmath")

add_subdirectory("${LLVM_MONOREPO_DIR}/runtimes" "runtimes")
