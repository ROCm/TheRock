# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# On Linux, we compile core libraries with our own LLVM. This gets us the
# best optimizations and, most importantly, gives us uniform access to
# sanitizers. Currently, some of these projects, though, are hard-coded
# for MSVC on Windows, so on that platform, just build them with the system
# compiler.
# TODO: Switch windows to use our own clang.
set(_system_toolchain "amd-llvm")
if(WIN32)
  set(_system_toolchain "")
endif()

if(THEROCK_ENABLE_CORE_RUNTIME)
  ##############################################################################
  # ROCR-Runtime
  ##############################################################################

  therock_cmake_subproject_declare(ROCR-Runtime
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocr-runtime"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ROCR-Runtime"
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN
      "${_system_toolchain}"
    CMAKE_ARGS
      "-DBUILD_SHARED_LIBS=ON"
    BUILD_DEPS
      amd-llvm
      therock-simde
    RUNTIME_DEPS
      rocprofiler-register
      ${THEROCK_BUNDLED_ELFUTILS}
      ${THEROCK_BUNDLED_LIBDRM}
      ${THEROCK_BUNDLED_NUMACTL}
    INTERFACE_LINK_DIRS
      "lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib"
  )
  therock_cmake_subproject_glob_c_sources(ROCR-Runtime
    SUBDIRS
      libhsakmt
      runtime
  )
  therock_cmake_subproject_provide_package(ROCR-Runtime hsakmt lib/cmake/hsakmt)
  therock_cmake_subproject_provide_package(ROCR-Runtime hsa-runtime64 lib/cmake/hsa-runtime64)
  therock_cmake_subproject_activate(ROCR-Runtime)

  therock_test_validate_shared_lib(
    PATH ROCR-Runtime/dist/lib
    LIB_NAMES libhsa-runtime64.so
  )

  ##############################################################################
  # rocminfo
  ##############################################################################

  therock_cmake_subproject_declare(rocminfo
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocminfo"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocminfo"
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN
      "${_system_toolchain}"
    RUNTIME_DEPS
      ROCR-Runtime
  )
  therock_cmake_subproject_glob_c_sources(rocminfo SUBDIRS .)
  therock_cmake_subproject_activate(rocminfo)

  therock_provide_artifact(core-runtime
    TARGET_NEUTRAL
    DESCRIPTOR artifact-core-runtime.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      ROCR-Runtime
      rocminfo
  )

endif(THEROCK_ENABLE_CORE_RUNTIME)

if(THEROCK_ENABLE_COMPILER)
  ##############################################################################
  # amd-comgr
  # A client of libLLVM which provides an in-process compiler API to the HIP
  # runtime. LLVM/Clang/LLD are statically linked and hidden behind comgr's
  # version script, avoiding symbol interposition issues.
  ##############################################################################

  therock_cmake_subproject_declare(amd-comgr
    USE_DIST_AMDGPU_TARGETS
    NO_INSTALL_RPATH  # See manual handling in the pre_hook.
    EXTERNAL_SOURCE_DIR "amd-llvm/amd/comgr"
    BINARY_DIR "amd-comgr"
    BACKGROUND_BUILD
    CMAKE_ARGS
      # Statically link LLVM/Clang into comgr. Combined with comgr's
      # -fvisibility=hidden and version script (local: *;), this hides all LLVM
      # symbols inside libamd_comgr.so, preventing symbol interposition without
      # requiring namespace isolation via dlmopen.
      -DCOMGR_STATIC_LLVM=ON
      # TODO: Currently unstable. Enable in >6.4.
      -DCOMGR_DISABLE_SPIRV=ON
      -DTHEROCK_HIP_MAJOR_VERSION=${THEROCK_HIP_MAJOR_VERSION}
      -DTHEROCK_HIP_MINOR_VERSION=${THEROCK_HIP_MINOR_VERSION}
      "-DLLVM_VERSION_SUFFIX=-rocm${ROCM_MAJOR_VERSION}"
      -DLLVM_ENABLE_SYMBOL_VERSIONING=ON
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      amd-llvm
      ${THEROCK_BUNDLED_ZSTD}
    INTERFACE_LINK_DIRS
      "lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib"
  )
  therock_cmake_subproject_provide_package(amd-comgr amd_comgr lib/cmake/amd_comgr)
  therock_cmake_subproject_activate(amd-comgr)

  therock_test_validate_shared_lib(
    PATH amd-comgr/dist/lib
    LIB_NAMES
      libamd_comgr.so
  )

  ##############################################################################
  # hipcc
  # Provides hipcc and hipconfig
  ##############################################################################

  therock_cmake_subproject_declare(hipcc
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "amd-llvm/amd/hipcc"
    BINARY_DIR "hipcc"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIPCC_BACKWARD_COMPATIBILITY=OFF
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      amd-llvm
  )
  therock_cmake_subproject_activate(hipcc)

  therock_provide_artifact(amd-llvm
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-llvm.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      hipcc
      amd-comgr
  )

endif(THEROCK_ENABLE_COMPILER)

##############################################################################
# amd-llvm-flang
# Fortran compiler (flang) + MLIR, built separately from base LLVM.
# Gated by THEROCK_ENABLE_FLANG (Linux-only).
##############################################################################

if(THEROCK_ENABLE_FLANG)
  # Building Flang is very memory-intensive on systems with many cores. Allow
  # users to specify the level of concurrency for these tasks so they can
  # reduce the risk of running out of memory.
  set(_extra_flang_cmake_args)
  if(FLANG_PARALLEL_COMPILE_JOBS)
    message(STATUS "Flang build concurrency level set to ${FLANG_PARALLEL_COMPILE_JOBS}")
    list(APPEND _extra_flang_cmake_args "-DFLANG_PARALLEL_COMPILE_JOBS=${FLANG_PARALLEL_COMPILE_JOBS}")
  else()
    message(WARNING "Flang concurrency is unlimited. Memory usage may be much higher. Set -DFLANG_PARALLEL_COMPILE_JOBS to reduce memory use.")
  endif()
  if(LLVM_PARALLEL_LINK_JOBS)
    list(APPEND _extra_flang_cmake_args "-DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}")
  endif()

  therock_cmake_subproject_declare(amd-llvm-flang
    EXTERNAL_SOURCE_DIR "amd-llvm-flang"
    BINARY_DIR "amd-llvm-flang"
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN amd-llvm
    CMAKE_ARGS
      ${_extra_flang_cmake_args}
    RUNTIME_DEPS
      amd-llvm
    INTERFACE_PROGRAM_DIRS
      lib/llvm/bin
    INSTALL_DESTINATION "lib/llvm"
  )
  therock_cmake_subproject_provide_package(amd-llvm-flang Flang lib/llvm/lib/cmake/flang)
  therock_cmake_subproject_activate(amd-llvm-flang)

  therock_provide_artifact(amd-llvm-flang
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-llvm-flang.toml
    COMPONENTS
      dbg
      dev
      run
    SUBPROJECT_DEPS
      amd-llvm-flang
  )
endif()

##############################################################################
# amd-llvm-offload
# Offload runtimes (libomptarget, device RTLs) + conditionally flang-rt.
# Always built on non-Windows.
##############################################################################

if(THEROCK_ENABLE_AMD_LLVM_OFFLOAD)
  set(_offload_deps amd-llvm)
  set(_extra_offload_cmake_args)
  if(THEROCK_ENABLE_FLANG)
    list(APPEND _offload_deps amd-llvm-flang)
    # flang-rt needs the Fortran compiler. Since flang-rt doesn't exist yet,
    # skip CMake's compiler check. See LLVM/flang-rt/README.
    list(APPEND _extra_offload_cmake_args
      -DCMAKE_Fortran_COMPILER_WORKS=YES
      "-DCMAKE_Fortran_COMPILER=${CMAKE_CURRENT_BINARY_DIR}/amd-llvm-flang/dist/lib/llvm/bin/flang"
    )
  endif()

  therock_cmake_subproject_declare(amd-llvm-offload
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "amd-llvm-offload"
    BINARY_DIR "amd-llvm-offload"
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN amd-llvm
    CMAKE_ARGS
      -DTHEROCK_ENABLE_FLANG=${THEROCK_ENABLE_FLANG}
      ${_extra_offload_cmake_args}
    RUNTIME_DEPS
      ${_offload_deps}
      ROCR-Runtime
      ${THEROCK_BUNDLED_ZLIB}
      ${THEROCK_BUNDLED_ZSTD}
      ${THEROCK_BUNDLED_NUMACTL}
      ${THEROCK_BUNDLED_ELFUTILS}
      ${THEROCK_BUNDLED_LIBDRM}
    INSTALL_DESTINATION "lib/llvm"
  )
  therock_cmake_subproject_activate(amd-llvm-offload)

  therock_provide_artifact(amd-llvm-offload
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-llvm-offload.toml
    COMPONENTS
      dbg
      lib
      run
    SUBPROJECT_DEPS
      amd-llvm-offload
  )
endif()

if(THEROCK_ENABLE_HIPIFY)
  ##############################################################################
  # HIPIFY
  ##############################################################################

  therock_cmake_subproject_declare(hipify
    USE_DIST_AMDGPU_TARGETS
    NO_INSTALL_RPATH  # See manual handling in the pre_hook.
    EXTERNAL_SOURCE_DIR "hipify"
    BACKGROUND_BUILD
    INTERFACE_PROGRAM_DIRS
      bin
    CMAKE_ARGS
      -DHIPIFY_INSTALL_CLANG_HEADERS=OFF
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      amd-llvm
  )
  therock_cmake_subproject_glob_c_sources(hipify
    SUBDIRS
      src
  )
  therock_cmake_subproject_activate(hipify)

  therock_provide_artifact(hipify
    TARGET_NEUTRAL
    DESCRIPTOR artifact-hipify.toml
    COMPONENTS
      run
      dbg
    SUBPROJECT_DEPS
      hipify
  )
endif(THEROCK_ENABLE_HIPIFY)
