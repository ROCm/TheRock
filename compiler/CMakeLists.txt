# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
      amd-llvm
      hipcc
      amd-comgr
  )
endif(THEROCK_ENABLE_COMPILER)

##############################################################################
# amd-llvm-flang
# Fortran compiler (Flang) + MLIR, built separately from base LLVM.
##############################################################################

if(THEROCK_ENABLE_FLANG)
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
    DISABLE_AMDGPU_TARGETS
    BACKGROUND_BUILD
    CMAKE_ARGS
      ${_extra_flang_cmake_args}
    RUNTIME_DEPS
      amd-llvm
    INTERFACE_PROGRAM_DIRS
      lib/llvm/bin
    INSTALL_DESTINATION "lib/llvm"
  )
  therock_cmake_subproject_provide_package(amd-llvm-flang Flang lib/llvm/lib/cmake/flang)
  therock_cmake_subproject_provide_package(amd-llvm-flang MLIR lib/llvm/lib/cmake/mlir)
  therock_cmake_subproject_activate(amd-llvm-flang)

  therock_provide_artifact(amd-llvm-flang
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-llvm-flang.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      amd-llvm-flang
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
