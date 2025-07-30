if(THEROCK_ENABLE_COMPILER)
  ##############################################################################
  # amd-llvm
  # Everything built as part of LLVM installs to lib/llvm
  # This includes the device-libs, since they must be findable by the compiler
  # frontend.
  ##############################################################################

  set(_extra_llvm_cmake_args)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND _extra_llvm_cmake_args "-DLLVM_ENABLE_PEDANTIC=OFF")
  endif()

  # If the compiler is not pristine (i.e. has patches), then there will be a
  # ".amd-llvm.smrev" file present which must be used instead of the auto
  # computed git revision. If present, this will have a stable hash of
  # revision plus applied patches suitable for run to run compiler
  # fingerprinting.
  set(LLVM_SMREV_FILE "${CMAKE_CURRENT_SOURCE_DIR}/.amd-llvm.smrev")
  if(EXISTS "${LLVM_SMREV_FILE}")
    file(STRINGS "${LLVM_SMREV_FILE}" LLVM_SMREV_LINES)
    list(GET LLVM_SMREV_LINES 0 LLVM_SMREV_REPO)
    list(GET LLVM_SMREV_LINES 1 LLVM_SMREV_REVISION)
    message(STATUS "Using stable amd-llvm revision info: Repo='${LLVM_SMREV_REPO}', Revision='${LLVM_SMREV_REVISION}'")
  endif()

  therock_cmake_subproject_declare(amd-llvm
    USE_DIST_AMDGPU_TARGETS
    NO_INSTALL_RPATH  # See manual handling in the pre_hook.
    EXTERNAL_SOURCE_DIR "amd-llvm"
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
      -DLLVM_INCLUDE_TESTS=OFF
      -DLLVM_ENABLE_ZLIB=FORCE_ON
      -DLLVM_ENABLE_Z3_SOLVER=OFF
      -DLLVM_ENABLE_LIBXML2=OFF
      -DCLANG_DEFAULT_LINKER=lld
      -DCLANG_DEFAULT_RTLIB=compiler-rt
      -DCLANG_DEFAULT_UNWINDLIB=libgcc
      -DCLANG_ENABLE_AMDCLANG=ON
      -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF
      -DLLVM_BUILD_LLVM_C_DYLIB=OFF

      # Clang features.
      -DCLANG_ENABLE_ARCMT=OFF
      -DCLANG_ENABLE_STATIC_ANALYZER=OFF
      -DCLANG_ENABLE_CLANGD=OFF
      -DCLANG_TIDY_ENABLE_STATIC_ANALYZER=OFF

      ${_extra_llvm_cmake_args}
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      ${THEROCK_BUNDLED_ZLIB}
      ${THEROCK_BUNDLED_ZSTD}
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
  therock_cmake_subproject_activate(amd-llvm)


  ##############################################################################
  # comgr
  # A client of libLLVM which provides an in-process compiler API to the HIP
  # runtime.
  ##############################################################################

  therock_cmake_subproject_declare(amd-comgr
    USE_DIST_AMDGPU_TARGETS
    NO_INSTALL_RPATH  # See manual handling in the pre_hook.
    EXTERNAL_SOURCE_DIR "amd-llvm/amd/comgr"
    BINARY_DIR "amd-comgr"
    BACKGROUND_BUILD
    CMAKE_ARGS
      # TODO: Currently unstable. Enable in >6.4.
      -DCOMGR_DISABLE_SPIRV=ON
      -DTHEROCK_HIP_MAJOR_VERSION=${THEROCK_HIP_MAJOR_VERSION}
      -DTHEROCK_HIP_MINOR_VERSION=${THEROCK_HIP_MINOR_VERSION}
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      amd-llvm
      ${THEROCK_BUNDLED_ZSTD}
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
      amd-comgr
      hipcc
  )
endif(THEROCK_ENABLE_COMPILER)

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
