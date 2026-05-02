# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.20)
project(amd-llvm-offload C CXX ASM)

include(ExternalProject)

set(LLVM_MONOREPO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../amd-llvm")

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")

set(PACKAGE_VERSION ${LLVM_PACKAGE_VERSION})
set(LLVM_INCLUDE_TESTS OFF)

set(_runtimes "openmp;offload")
set(_amdgcn_runtimes "openmp")

if(THEROCK_ENABLE_FLANG)
  list(APPEND _runtimes "flang-rt")
  list(APPEND _amdgcn_runtimes "flang-rt")
  set(FLANG_RUNTIME_F128_MATH_LIB "libquadmath")
  set(LIBOMPTARGET_BUILD_DEVICE_FORTRT ON)
  set(LIBOMP_FORTRAN_MODULES ON)
  set(RUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBC_PROVIDER "llvm")
  set(RUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_LIBCXX_PROVIDER "llvm")
  set(RUNTIMES_amdgcn-amd-amdhsa_FLANG_RT_INCLUDE_AMD ON)
  set(LIBOMP_FORTRAN_MODULES_COMPILER "${CMAKE_Fortran_COMPILER}")
  set(LIBOMP_MODULES_INSTALL_PATH "${CMAKE_INSTALL_INCLUDEDIR}/flang")
else()
  set(LIBOMPTARGET_BUILD_DEVICE_FORTRT OFF)
endif()

set(LLVM_ENABLE_RUNTIMES "${_runtimes}" CACHE STRING "" FORCE)
set(CMAKE_DISABLE_FIND_PACKAGE_FFI TRUE)
set(OPENMP_ENABLE_LIBOMPTARGET ON)
set(LIBOMPTARGET_ENABLE_DEBUG ON)
set(LIBOMPTARGET_NO_SANITIZER_AMDGPU ON)
set(RUNTIMES_CMAKE_ARGS "-DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON")

set(LLVM_RUNTIME_TARGETS "default;amdgcn-amd-amdhsa")
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON)
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES "${_amdgcn_runtimes}")
set(LIBOMP_COPY_EXPORTS OFF)

add_subdirectory("${LLVM_MONOREPO_DIR}/runtimes" "runtimes")

find_package(Clang REQUIRED CONFIG)
find_package(AMDDeviceLibs REQUIRED CONFIG)

set(_openmp_device_build_dir "${CMAKE_CURRENT_BINARY_DIR}/openmp-device")
set(_openmp_device_install_dir "${CMAKE_INSTALL_PREFIX}")
set(_openmp_device_cmake_args
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
  "-DLLVM_ENABLE_RUNTIMES=openmp"
  "-DCMAKE_C_COMPILER=${LLVM_TOOLS_BINARY_DIR}/clang"
  "-DCMAKE_CXX_COMPILER=${LLVM_TOOLS_BINARY_DIR}/clang++"
  "-DCMAKE_C_COMPILER_TARGET=amdgcn-amd-amdhsa"
  "-DCMAKE_CXX_COMPILER_TARGET=amdgcn-amd-amdhsa"
  "-DLLVM_DEFAULT_TARGET_TRIPLE=amdgcn-amd-amdhsa"
  "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"
  "-DLLVM_DIR=${LLVM_DIR}"
  "-DClang_DIR=${Clang_DIR}"
  "-DAMDDeviceLibs_DIR=${AMDDeviceLibs_DIR}"
  "-DCMAKE_C_COMPILER_WORKS=ON"
  "-DCMAKE_CXX_COMPILER_WORKS=ON"
  "-DCMAKE_ASM_COMPILER_WORKS=ON"
  "-DCMAKE_DISABLE_FIND_PACKAGE_LibEdit=ON"
  "-DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON"
  "-DCMAKE_DISABLE_FIND_PACKAGE_zstd=ON"
)
ExternalProject_Add(openmp-device-runtime
  SOURCE_DIR "${LLVM_MONOREPO_DIR}/runtimes"
  BINARY_DIR "${_openmp_device_build_dir}"
  INSTALL_DIR "${_openmp_device_install_dir}"
  CMAKE_ARGS ${_openmp_device_cmake_args}
  BUILD_BYPRODUCTS
    "${_openmp_device_install_dir}/lib/libompdevice.a"
    "${_openmp_device_install_dir}/lib/libomptarget-amdgpu.bc"
)
install(CODE "
  set(_openmp_device_resource_dir \"\${CMAKE_INSTALL_PREFIX}/lib/clang/${LLVM_VERSION_MAJOR}/lib/amdgcn-amd-amdhsa\")
  file(MAKE_DIRECTORY \"\${_openmp_device_resource_dir}\")
  file(COPY
    \"\${CMAKE_INSTALL_PREFIX}/lib/libompdevice.a\"
    \"\${CMAKE_INSTALL_PREFIX}/lib/libomptarget-amdgpu.bc\"
    DESTINATION \"\${_openmp_device_resource_dir}\")
")
