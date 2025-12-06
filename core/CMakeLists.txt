if(THEROCK_ENABLE_CORE_RUNTIME)
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

if(THEROCK_ENABLE_HIP_RUNTIME)
  ##############################################################################
  # clr
  # The primary HIP compiler and runtime target. This is also the project used
  # to root the "amd-hip" compiler toolchain and is a superset of amd-llvm.
  # As a particularly sharp edge, the `hipconfig` tool is really only fully
  # functional from this project (which contributes version and metadata).
  ##############################################################################
  set(HIP_CLR_CMAKE_ARGS)
  set(HIP_CLR_RUNTIME_DEPS)
  if(WIN32)
    # Windows CLR options
    set(_compute_pal_dir "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/amdgpu-windows-interop")
    cmake_path(NORMAL_PATH _compute_pal_dir)
    list(APPEND HIP_CLR_CMAKE_ARGS
      "-DUSE_PROF_API=OFF"
      "-D__HIP_ENABLE_PCH=OFF"
      "-DROCCLR_ENABLE_PAL=1"
      "-DROCCLR_ENABLE_HSA=0"
      "-DAMD_COMPUTE_WIN=${_compute_pal_dir}"
    )
  else()
    # Non-Windows CLR options.
    list(APPEND HIP_CLR_RUNTIME_DEPS
      rocminfo  # Various things expect to be able to find the rocminfo tools
      rocprofiler-register
      ROCR-Runtime
    )
  endif()

  therock_cmake_subproject_declare(hip-clr
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/clr"
    INTERFACE_PROGRAM_DIRS
      bin
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN
      "${_system_toolchain}"
    CMAKE_ARGS
      "-DHIP_PLATFORM=amd"
      "-DHIP_COMMON_DIR=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip"
      "-DCLR_BUILD_HIP=ON"
      # Legacy: Disable various auto-detection logic that breaks out of jail
      # and can use local machine tools.
      "-DHIPCC_BIN_DIR="
      ${HIP_CLR_CMAKE_ARGS}
    BUILD_DEPS
      rocm-cmake
      therock-simde
    RUNTIME_DEPS
      amd-llvm
      amd-comgr
      hipcc     # For hipconfig
      rocm-core
      ${HIP_CLR_RUNTIME_DEPS}
    INTERFACE_LINK_DIRS
      "lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib"
  )
  therock_cmake_subproject_glob_c_sources(hip-clr SUBDIRS .)
  therock_cmake_subproject_provide_package(hip-clr hip lib/cmake/hip)
  # TODO: Some projects resolve "hip" vs "HIP" so we advertise both, but this isn't
  # great.
  therock_cmake_subproject_provide_package(hip-clr HIP lib/cmake/hip)
  therock_cmake_subproject_provide_package(hip-clr hip-lang lib/cmake/hip-lang)
  therock_cmake_subproject_provide_package(hip-clr hiprtc lib/cmake/hiprtc)
  therock_cmake_subproject_activate(hip-clr)

  therock_provide_artifact(core-hip
    TARGET_NEUTRAL
    DESCRIPTOR artifact-core-hip.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      hip-clr
  )

  therock_test_validate_shared_lib(
    PATH clr/dist/lib
    LIB_NAMES
      libamdhip64.so
      libhiprtc-builtins.so
      libhiprtc.so
  )
  if(WIN32)
    ##############################################################################
    # hipInfo (from hip-tests)
    ##############################################################################
    therock_cmake_subproject_declare(hipInfo
      EXTERNAL_SOURCE_DIR "hipInfo"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipInfo"
      BACKGROUND_BUILD
      COMPILER_TOOLCHAIN
        "${_system_toolchain}"
      CMAKE_ARGS
        "-DHIP_PLATFORM=amd"
        "-DHIP_TESTS_DIR=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip-tests"
      RUNTIME_DEPS
        hip-clr
    )
    therock_cmake_subproject_glob_c_sources(hipInfo SUBDIRS .)
    therock_cmake_subproject_activate(hipInfo)

    # Provide it as an artifact:
    therock_provide_artifact(core-hipinfo
      TARGET_NEUTRAL
      DESCRIPTOR artifact-core-hipinfo.toml
      COMPONENTS
        run
      SUBPROJECT_DEPS
        hipInfo
    )
  endif()

  if(THEROCK_BUILD_TESTING)
    set(HIP_TESTS_CMAKE_ARGS)
    set(_hip_tests_subproject_names)
    therock_cmake_subproject_declare(hip-tests
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hip-tests/catch"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip-tests"
      BACKGROUND_BUILD
      COMPILER_TOOLCHAIN
        amd-hip
      CMAKE_ARGS
        -DHIP_PLATFORM="amd"
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
      BUILD_DEPS
        rocm-cmake
        therock-boost
        therock-catch2
      RUNTIME_DEPS
        hip-clr
        ${THEROCK_BUNDLED_NUMACTL}
      INTERFACE_LINK_DIRS
        lib
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib
        lib/rocm_sysdeps/lib
    )
    therock_cmake_subproject_glob_c_sources(hip-tests SUBDIRS catch)
    therock_cmake_subproject_activate(hip-tests)
    if(WIN32)
      set(hip-tests_artifact_file artifact-core-hiptests_win.toml)
    else()
      set(hip-tests_artifact_file artifact-core-hiptests.toml)
    endif()
    # Provide it as an artifact:
    therock_provide_artifact(core-hiptests
      TARGETS USE_DIST_AMDGPU_TARGETS
      DESCRIPTOR ${hip-tests_artifact_file}
      COMPONENTS
        test
      SUBPROJECT_DEPS
        hip-tests
    )
    list(APPEND _hip_tests_subproject_names hip-tests)
  endif(THEROCK_BUILD_TESTING)
endif(THEROCK_ENABLE_HIP_RUNTIME)

################################################################################
# ocl-icd
################################################################################

therock_cmake_subproject_declare(ocl-icd
  USE_DIST_AMDGPU_TARGETS
  EXTERNAL_SOURCE_DIR
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr/opencl/khronos/icd"
    "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/OpenCL-Headers"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/opencl/icd"
  BACKGROUND_BUILD
  CMAKE_ARGS
    "-DOPENCL_ICD_LOADER_REQUIRE_WDK=OFF"
    "-DOPENCL_ICD_LOADER_HEADERS_DIR=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr/opencl/khronos/headers/opencl2.2"

  INTERFACE_LINK_DIRS
    "lib"
  INTERFACE_INSTALL_RPATH_DIRS
    "lib"
)
therock_cmake_subproject_activate(ocl-icd)

if(THEROCK_ENABLE_OCL_RUNTIME)
  set(OCL_CLR_CMAKE_ARGS)
  set(OCL_CLR_RUNTIME_DEPS)
  if(WIN32)
    # Windows CLR options
    set(_compute_pal_dir "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/amdgpu-windows-interop")
    cmake_path(NORMAL_PATH _compute_pal_dir)
    list(APPEND OCL_CLR_CMAKE_ARGS
      "-DUSE_PROF_API=OFF"
      "-D__HIP_ENABLE_PCH=OFF"
      "-DROCCLR_ENABLE_PAL=ON"
      "-DROCCLR_ENABLE_HSA=OFF"
      "-DROCCLR_ENABLE_HSAIL=ON"
      "-DROCCLR_ENABLE_LC=ON"
      "-DAMD_COMPUTE_WIN=${_compute_pal_dir}"
    )
  else()
    # Non-Windows CLR options.
    list(APPEND OCL_CLR_RUNTIME_DEPS
      rocminfo  # Various things expect to be able to find the rocminfo tools
      rocprofiler-register
      ROCR-Runtime
    )
  endif()

  therock_cmake_subproject_declare(ocl-clr
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/clr"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ocl-clr"
    INTERFACE_PROGRAM_DIRS
      bin
    BACKGROUND_BUILD
    CMAKE_ARGS
      "-DCLR_BUILD_OCL=ON"
      ${OCL_CLR_CMAKE_ARGS}
    BUILD_DEPS
      rocm-cmake
    RUNTIME_DEPS
      amd-llvm
      amd-comgr
      rocm-core
      ocl-icd
      ${OCL_CLR_RUNTIME_DEPS}
    INTERFACE_LINK_DIRS
      "lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib"
  )
  therock_cmake_subproject_glob_c_sources(ocl-clr SUBDIRS .)
  therock_cmake_subproject_provide_package(ocl-clr ocl-icd lib/cmake/ocl-icd)
  therock_cmake_subproject_activate(ocl-clr)

  therock_provide_artifact(core-ocl
    TARGET_NEUTRAL
    DESCRIPTOR artifact-core-ocl.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      ocl-clr
  )

  therock_test_validate_shared_lib(
    PATH ocl-clr/dist/lib/opencl
    LIB_NAMES
      libamdocl64.so
  )

  # Ensure runtime dependencies of libamdocl64.so (for example,
  # libamd_comgr_loader.so) are discoverable by the dynamic loader when
  # running the validation test. Some configurations (e.g. Windows) do not
  # register this test, so guard the property update accordingly.
  if(TEST therock-validate-shared-lib-libamdocl64.so)
    set_tests_properties(therock-validate-shared-lib-libamdocl64.so
      PROPERTIES
        ENVIRONMENT
          "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/core/ocl-clr/dist/lib:$ENV{LD_LIBRARY_PATH}"
    )
  endif()

endif(THEROCK_ENABLE_OCL_RUNTIME)
