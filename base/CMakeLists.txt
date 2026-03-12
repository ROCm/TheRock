# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Optional subprojects add themselves to this list for artifact dependencies.
set(_optional_artifact_deps)

################################################################################
# aux-overlay
################################################################################

therock_cmake_subproject_declare(therock-aux-overlay
  FPRINT_SOURCE_HASH
  EXTERNAL_SOURCE_DIR "aux-overlay"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  CMAKE_ARGS
    "-DTHEROCK_FLAG_SETTINGS_FILE=${THEROCK_FLAG_SETTINGS_FILE}"
)
therock_cmake_subproject_activate(therock-aux-overlay)


################################################################################
# rocm-cmake
################################################################################

therock_cmake_subproject_declare(rocm-cmake
  EXTERNAL_SOURCE_DIR "rocm-cmake"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
)
therock_cmake_subproject_provide_package(rocm-cmake
  ROCmCMakeBuildTools share/rocmcmakebuildtools/cmake)
therock_cmake_subproject_provide_package(rocm-cmake
  ROCM share/rocm/cmake)
therock_cmake_subproject_activate(rocm-cmake)


################################################################################
# rocm-core
################################################################################

if(WIN32)
  set(_shared_libs_arg "OFF")
else()
  set(_shared_libs_arg "ON")
endif()

therock_cmake_subproject_declare(rocm-core
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocm-core"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocm-core"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  CMAKE_ARGS
    "-DBUILD_SHARED_LIBS=${_shared_libs_arg}"
    "-DROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
)
therock_cmake_subproject_glob_c_sources(rocm-core
  SUBDIRS .
)
therock_cmake_subproject_provide_package(rocm-core rocm-core lib/cmake/rocm-core)
therock_cmake_subproject_activate(rocm-core)



################################################################################
# rocm_smi_lib
################################################################################

if(NOT WIN32)  # TODO(#36): Enable on Windows and/or make subproject inclusion generally optional

therock_cmake_subproject_declare(rocm_smi_lib
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocm-smi-lib"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocm_smi_lib"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  CMAKE_ARGS
    # See the post_hook, which needs to advertise install interface directories
    # in this case.
    "-DTHEROCK_HAS_BUNDLED_LIBDRM=${THEROCK_BUNDLED_LIBDRM}"
  INTERFACE_LINK_DIRS
    "lib"
  INSTALL_RPATH_DIRS
    "lib"
  INTERFACE_INSTALL_RPATH_DIRS
    "lib"
  INTERFACE_PROGRAM_DIRS
    "bin"
  RUNTIME_DEPS
    ${THEROCK_BUNDLED_LIBDRM}
)
therock_cmake_subproject_glob_c_sources(rocm_smi_lib
  SUBDIRS
    include
    src
)
therock_cmake_subproject_provide_package(rocm_smi_lib rocm_smi lib/cmake/rocm_smi)
therock_cmake_subproject_activate(rocm_smi_lib)
list(APPEND _optional_artifact_deps rocm_smi_lib)

endif()


################################################################################
# rocprofiler-register
# This is a stub that helps runtime libraries and profiles rendezvous
################################################################################

if(NOT WIN32)  # TODO(#36): Enable on Windows?

therock_cmake_subproject_declare(rocprofiler-register
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprofiler-register"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocprofiler-register"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  INTERFACE_LINK_DIRS
    "lib"
  INTERFACE_INSTALL_RPATH_DIRS
    "lib"
)
therock_cmake_subproject_glob_c_sources(rocprofiler-register
  SUBDIRS
    source/include
    source/lib
)
therock_cmake_subproject_provide_package(rocprofiler-register
  rocprofiler-register lib/cmake/rocprofiler-register)
therock_cmake_subproject_activate(rocprofiler-register)
list(APPEND _optional_artifact_deps rocprofiler-register)

endif()

################################################################################
# rocm-half
################################################################################

therock_cmake_subproject_declare(rocm-half
  EXTERNAL_SOURCE_DIR "half"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  INTERFACE_INCLUDE_DIRS
    # No find_package support (just a naked include).
    include
  BUILD_DEPS
    rocm-cmake
)
therock_cmake_subproject_glob_c_sources(rocm-half
  SUBDIRS
    include
)
therock_cmake_subproject_activate(rocm-half)


################################################################################
# rocm-kpack
################################################################################

if(THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS)

therock_cmake_subproject_declare(rocm-kpack
  EXTERNAL_SOURCE_DIR "rocm-kpack"
  USE_DIST_AMDGPU_TARGETS
  BACKGROUND_BUILD
  OPTIONAL
  CMAKE_ARGS
    -DBUILD_TESTING=OFF
    -DBUILD_SHARED_LIBS=ON
  BUILD_DEPS
    therock-msgpack-cxx
  RUNTIME_DEPS
    ${THEROCK_BUNDLED_ZSTD}
  INTERFACE_LINK_DIRS
    lib
  INTERFACE_INSTALL_RPATH_DIRS
    lib
)
therock_cmake_subproject_provide_package(rocm-kpack
  rocm-kpack lib/cmake/rocm-kpack)
therock_cmake_subproject_activate(rocm-kpack)
list(APPEND _optional_artifact_deps rocm-kpack)

endif()


################################################################################
# amd-llvm
# Everything built as part of LLVM installs to lib/llvm
# This includes the device-libs, since they must be findable by the compiler
# frontend.
# Note: The amd-llvm sub-project declaration lives in base/ for ordering
# (it must be available before third-party and compiler), but the git
# submodule remains at compiler/amd-llvm/.
################################################################################

if(THEROCK_ENABLE_AMD_LLVM_BASE)

  set(_extra_llvm_cmake_args)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND _extra_llvm_cmake_args "-DLLVM_ENABLE_PEDANTIC=OFF")
  endif()

  # LLVM linking is very memory intensive. Allow users to adjust the number
  # of parallel link jobs.
  if(LLVM_PARALLEL_LINK_JOBS)
    message(STATUS "LLVM parallel link jobs set to ${LLVM_PARALLEL_LINK_JOBS}")
    list(APPEND _extra_llvm_cmake_args "-DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}")
  else()
     message(WARNING "LLVM parallel link jobs is unlimited. Memory usage may be much higher. Set -DLLVM_PARALLEL_LINK_JOBS to reduce memory use.")
  endif()

  # If the compiler is not pristine (i.e. has patches), then there will be a
  # ".amd-llvm.smrev" file present which must be used instead of the auto
  # computed git revision. If present, this will have a stable hash of
  # revision plus applied patches suitable for run to run compiler
  # fingerprinting.
  set(LLVM_SMREV_FILE "${THEROCK_SOURCE_DIR}/compiler/.amd-llvm.smrev")
  if(EXISTS "${LLVM_SMREV_FILE}")
    file(STRINGS "${LLVM_SMREV_FILE}" LLVM_SMREV_LINES)
    list(GET LLVM_SMREV_LINES 0 LLVM_SMREV_REPO)
    list(GET LLVM_SMREV_LINES 1 LLVM_SMREV_REVISION)
    message(STATUS "Using stable amd-llvm revision info: Repo='${LLVM_SMREV_REPO}', Revision='${LLVM_SMREV_REVISION}'")
  endif()

  therock_cmake_subproject_declare(amd-llvm
    USE_DIST_AMDGPU_TARGETS
    NO_INSTALL_RPATH  # See manual handling in the pre_hook.
    EXTERNAL_SOURCE_DIR "${THEROCK_SOURCE_DIR}/compiler/amd-llvm"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/amd-llvm"
    # Note that LLVM top level CMakeLists.txt is in the llvm subdir of the
    # monorepo.
    CMAKE_LISTS_RELPATH "llvm"
    INTERFACE_PROGRAM_DIRS
      lib/llvm/bin
    CMAKE_ARGS
      # Version configuration
      # Note that if LLVM_SMREV_REPO and LLVM_FORCE_VC_REVISION are empty,
      # LLVM will compute the tip of tree revision, which is appropriate for
      # pristine checkouts.
      -DLLVM_APPEND_VC_REV=ON
      "-DLLVM_FORCE_VC_REPOSITORY=${LLVM_SMREV_REPO}"
      "-DLLVM_FORCE_VC_REVISION=${LLVM_SMREV_REVISION}"

      # LIBCXX
      -DLIBCXX_ENABLE_SHARED=OFF
      -DLIBCXX_ENABLE_STATIC=ON
      -DLIBCXX_INSTALL_LIBRARY=OFF
      -DLIBCXX_INSTALL_HEADERS=OFF
      -DLIBCXXABI_ENABLE_SHARED=OFF
      -DLIBCXXABI_ENABLE_STATIC=ON

      # Features
      -DLLVM_INCLUDE_TESTS=${THEROCK_ENABLE_LLVM_TESTS}
      -DLLVM_ENABLE_ZLIB=FORCE_ON
      -DLLVM_ENABLE_Z3_SOLVER=OFF
      -DLLVM_ENABLE_LIBXML2=OFF
      -DCLANG_DEFAULT_LINKER=lld
      -DCLANG_DEFAULT_RTLIB=compiler-rt
      -DCLANG_DEFAULT_UNWINDLIB=libgcc
      -DCLANG_ENABLE_AMDCLANG=ON
      -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF
      -DLLVM_BUILD_LLVM_C_DYLIB=OFF

      # Build optimizations
      -DLLVM_BUILD_EXAMPLES=${THEROCK_ENABLE_LLVM_TESTS}
      -DLLVM_INCLUDE_EXAMPLES=${THEROCK_ENABLE_LLVM_TESTS}
      -DLLVM_TOOL_LLVM_JITLINK_BUILD=${THEROCK_ENABLE_LLVM_TESTS}

      # Clang features.
      -DCLANG_ENABLE_ARCMT=OFF
      -DCLANG_ENABLE_STATIC_ANALYZER=OFF
      -DCLANG_ENABLE_CLANGD=OFF
      -DCLANG_TIDY_ENABLE_STATIC_ANALYZER=OFF

      # ASAN features.
      # Note that building AMDGPU ASAN extensions depends on various runtime headers
      # (since it instruments them). These paths must line up and will be an error
      # in the compiler-rt build if not found.
      "-DSANITIZER_AMDGPU=${THEROCK_CONDITION_IS_NON_WINDOWS}"
      "-DSANITIZER_HSA_INCLUDE_PATH=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocr-runtime/runtime/hsa-runtime/inc"
      "-DSANITIZER_COMGR_INCLUDE_PATH=${THEROCK_SOURCE_DIR}/compiler/amd-llvm/amd/comgr/include"

      # compiler-rt debug mode
      "-DCOMPILER_RT_DEBUG=${THEROCK_COMPILER_RT_DEBUG}"

      ${_extra_llvm_cmake_args}
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      ${THEROCK_BUNDLED_ZLIB}
      ${THEROCK_BUNDLED_ZSTD}
      ${THEROCK_BUNDLED_NUMACTL}
      ${THEROCK_BUNDLED_ELFUTILS}
      ${THEROCK_BUNDLED_LIBDRM}
    # The entire LLVM install tree is placed inside of the overall ROCM lib/llvm
    # directory.
    INSTALL_DESTINATION "lib/llvm"
    INTERFACE_LINK_DIRS
      "lib/llvm/lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib/llvm/lib"
  )
  # Note that we do not trigger on LLVM source changes because the monorepo is
  # too large to glob like that. Consider having a project dev mode option for
  # enabling better ergonomics here.

  therock_cmake_subproject_provide_package(amd-llvm AMDDeviceLibs lib/llvm/lib/cmake/AMDDeviceLibs)
  therock_cmake_subproject_provide_package(amd-llvm Clang lib/llvm/lib/cmake/clang)
  therock_cmake_subproject_provide_package(amd-llvm LLD lib/llvm/lib/cmake/lld)
  therock_cmake_subproject_provide_package(amd-llvm LLVM lib/llvm/lib/cmake/llvm)
  therock_cmake_subproject_provide_package(amd-llvm OpenMP lib/llvm/lib/cmake/openmp)
  therock_cmake_subproject_activate(amd-llvm)

  therock_provide_artifact(amd-llvm-base
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-llvm-base.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      amd-llvm
  )

endif(THEROCK_ENABLE_AMD_LLVM_BASE)


################################################################################
# Artifacts
################################################################################

therock_provide_artifact(base
  TARGET_NEUTRAL
  DESCRIPTOR artifact.toml
  COMPONENTS
    dbg
    dev
    doc
    lib
    run
    test
  SUBPROJECT_DEPS
    ${_optional_artifact_deps}
    rocm-cmake
    rocm-core
    rocm-half
    therock-aux-overlay
)
