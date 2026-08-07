# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Most libraries need to depend on the profiler, but is conditionally only used
# on Posix.
set(optional_profiler_deps)
if(THEROCK_FLAG_INCLUDE_PROFILER)
  list(APPEND optional_profiler_deps roctracer rocprofiler-sdk)
endif()

if(THEROCK_ENABLE_MIOPEN)
  # MIOpen's CK use is gated on both the per-arch CK availability
  # (math-libs/CMakeLists.txt) and the user-facing
  # THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL opt-out, which lets
  # someone build CK for other consumers while keeping it out of MIOpen.
  set(_miopen_use_ck OFF)
  if(_ck_enabled AND THEROCK_MIOPEN_USE_COMPOSABLE_KERNEL)
    set(_miopen_use_ck ON)
  endif()

  set(optional_miopen_build_deps)
  if(_miopen_use_ck)
    list(APPEND optional_miopen_build_deps composable_kernel)
  endif()

  if(MSVC)
    set(_THEROCK_MIOPEN_ENABLE_TESTING ON)
  else()
    set(_THEROCK_MIOPEN_ENABLE_TESTING ${THEROCK_BUILD_TESTING})
  endif()

  therock_cmake_subproject_declare(MIOpen
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/miopen"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/MIOpen"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DBUILD_TESTING=${_THEROCK_MIOPEN_ENABLE_TESTING}"
      "-DMIOPEN_USE_COMPOSABLEKERNEL=${_miopen_use_ck}"
      -DMIOPEN_USE_MLIR=OFF # TODO: enable
      -DMIOPEN_TEST_DISCRETE=OFF
      "-DMIOPEN_INSTALL_GPU_DATABASES=\"${THEROCK_AMDGPU_TARGETS}\""
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-eigen
      therock-frugally-deep
      therock-googletest
      therock-nlohmann-json
      therock-FunctionalPlus
      ${optional_miopen_build_deps}
    RUNTIME_DEPS
      hipBLAS-common
      hip-clr
      hipBLAS
      hipBLASLt
      rocm-half
      rocBLAS
      rocRAND
      ${THEROCK_BUNDLED_BZIP2}
      ${THEROCK_BUNDLED_SQLITE3}
      ${optional_profiler_deps}
    TEST_SUBPROJECTS
      miopenprovider
      hipDNN_samples
      hipdnn_integration_tests
  )
  therock_cmake_subproject_provide_package(MIOpen miopen lib/cmake/miopen)
  therock_cmake_subproject_activate(MIOpen)

  therock_provide_artifact(miopen
    DESCRIPTOR artifact-miopen.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      MIOpen
  )
endif(THEROCK_ENABLE_MIOPEN)

if(THEROCK_ENABLE_HIPDNN)
  ##############################################################################
  # hipDNN
  ##############################################################################

  therock_cmake_subproject_declare(hipDNN
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/hipdnn"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipDNN"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DHIP_DNN_SKIP_TESTS=$<NOT:$<BOOL:${THEROCK_BUILD_TESTING}>>"
      -DHIP_DNN_BUILD_PLUGINS=OFF
      -DHIP_DNN_GENERATE_SDK_HEADERS=OFF
      -DENABLE_CLANG_TIDY=OFF
      -DENABLE_CLANG_FORMAT=OFF
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      therock-googletest
      therock-fmt
      therock-spdlog
      therock-flatbuffers
      therock-nlohmann-json
    RUNTIME_DEPS
      hip-clr
    TEST_SUBPROJECTS
      miopenprovider
      hipblasltprovider
      hipkernelprovider
      hipdnn_integration_tests
  )
  therock_cmake_subproject_glob_c_sources(hipDNN
  SUBDIRS
    .
  )

  therock_cmake_subproject_provide_package(hipDNN hipdnn_data_sdk lib/cmake/hipdnn_data_sdk)
  therock_cmake_subproject_provide_package(hipDNN hipdnn_flatbuffers_sdk lib/cmake/hipdnn_flatbuffers_sdk)
  therock_cmake_subproject_provide_package(hipDNN hipdnn_plugin_sdk lib/cmake/hipdnn_plugin_sdk)
  therock_cmake_subproject_provide_package(hipDNN hipdnn_test_sdk lib/cmake/hipdnn_test_sdk)
  therock_cmake_subproject_provide_package(hipDNN hipdnn_frontend lib/cmake/hipdnn_frontend)
  therock_cmake_subproject_provide_package(hipDNN hipdnn_backend lib/cmake/hipdnn_backend)
  therock_cmake_subproject_activate(hipDNN)

  therock_provide_artifact(hipdnn
    TARGET_NEUTRAL
    DESCRIPTOR artifact-hipdnn.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      hipDNN
  )
endif(THEROCK_ENABLE_HIPDNN)

if(THEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS)
  ##############################################################################
  # hipDNN integration tests
  ##############################################################################

  therock_cmake_subproject_declare(hipdnn_integration_tests
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/dnn-providers/integration-tests"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipdnn_integration_tests"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DENABLE_CLANG_TIDY=OFF
      -DENABLE_CLANG_FORMAT=OFF
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      therock-googletest
      therock-flatbuffers
      therock-nlohmann-json
    RUNTIME_DEPS
      hip-clr
      hipDNN
    TEST_SUBPROJECTS
      miopenprovider
  )
  therock_cmake_subproject_glob_c_sources(hipdnn_integration_tests
  SUBDIRS
    .
  )

  therock_cmake_subproject_provide_package(hipdnn_integration_tests hipdnn_integration_tests lib/cmake/hipdnn_integration_tests)

  therock_cmake_subproject_activate(hipdnn_integration_tests)

  therock_provide_artifact(hipdnn-integration-tests
    TARGET_NEUTRAL
    DESCRIPTOR artifact-hipdnn-integration-tests.toml
    COMPONENTS
      dbg
      run
      test
    SUBPROJECT_DEPS
      hipdnn_integration_tests
  )

endif(THEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS)

if(THEROCK_ENABLE_MIOPENPROVIDER)
  ##############################################################################
  # hipDNN MIOpen provider
  ##############################################################################

  therock_cmake_subproject_declare(miopenprovider
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/dnn-providers/miopen-provider"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/miopenprovider"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DHIP_DNN_SKIP_TESTS=$<NOT:$<BOOL:${THEROCK_BUILD_TESTING}>>"
      -DENABLE_CLANG_TIDY=OFF
      -DENABLE_CLANG_FORMAT=OFF
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipDNN
      therock-googletest
      therock-flatbuffers
      therock-nlohmann-json
    RUNTIME_DEPS
      hipdnn_integration_tests
      MIOpen
    TEST_SUBPROJECTS
      hipDNN_samples
  )
  therock_cmake_subproject_glob_c_sources(miopenprovider
  SUBDIRS
    .
  )
  therock_cmake_subproject_activate(miopenprovider)

  therock_provide_artifact(miopenprovider
    TARGET_NEUTRAL
    DESCRIPTOR artifact-miopenprovider.toml
    COMPONENTS
      dbg
      lib
      test
    SUBPROJECT_DEPS
      miopenprovider
  )
endif(THEROCK_ENABLE_MIOPENPROVIDER)

if(THEROCK_ENABLE_HIPBLASLTPROVIDER)
  ##############################################################################
  # hipDNN hipBLASLt provider
  ##############################################################################

  therock_cmake_subproject_declare(hipblasltprovider
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/dnn-providers/hipblaslt-provider"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipblasltprovider"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DHIP_DNN_SKIP_TESTS=$<NOT:$<BOOL:${THEROCK_BUILD_TESTING}>>"
      -DENABLE_CLANG_TIDY=OFF
      -DENABLE_CLANG_FORMAT=OFF
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipDNN
      therock-googletest
      therock-flatbuffers
      therock-nlohmann-json
    RUNTIME_DEPS
      hipBLAS-common
      hipBLASLt
      hipdnn_integration_tests
    TEST_SUBPROJECTS
  )
  therock_cmake_subproject_glob_c_sources(hipblasltprovider
  SUBDIRS
    .
  )
  therock_cmake_subproject_activate(hipblasltprovider)

  therock_provide_artifact(hipblasltprovider
    TARGET_NEUTRAL
    DESCRIPTOR artifact-hipblasltprovider.toml
    COMPONENTS
      dbg
      lib
      test
    SUBPROJECT_DEPS
      hipblasltprovider
  )
endif(THEROCK_ENABLE_HIPBLASLTPROVIDER)

if(THEROCK_ENABLE_HIPKERNELPROVIDER)
  ##############################################################################
  # hipDNN hip kernel provider
  ##############################################################################
  therock_cmake_subproject_declare(hipkernelprovider
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/dnn-providers/hip-kernel-provider"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipkernelprovider"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DHIPKERNELPROVIDER_ENABLE_TESTS=${THEROCK_BUILD_TESTING}
      -DENABLE_CLANG_TIDY=OFF
      -DENABLE_CLANG_FORMAT=OFF
      -DROCKE_KPACK_PYTHON_DIR=${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/shared/kpack/python
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipDNN
      therock-googletest
      therock-flatbuffers
      therock-nlohmann-json
      amd-comgr
    RUNTIME_DEPS
      hip-clr
      hipdnn_integration_tests
      rocm-kpack
    TEST_SUBPROJECTS
  )
  therock_cmake_subproject_glob_c_sources(hipkernelprovider
  SUBDIRS
    .
  )
  therock_cmake_subproject_activate(hipkernelprovider)

  therock_provide_artifact(hipkernelprovider
    DESCRIPTOR artifact-hipkernelprovider.toml
    COMPONENTS
      dbg
      lib
      test
    SUBPROJECT_DEPS
      hipkernelprovider
  )
endif(THEROCK_ENABLE_HIPKERNELPROVIDER)

if(THEROCK_ENABLE_HIPDNN_SAMPLES)
  ##############################################################################
  # hipDNN samples
  ##############################################################################

  therock_cmake_subproject_declare(hipDNN_samples
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/projects/hipdnn/samples"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipdnn_samples"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DHIP_DNN_SKIP_TESTS=$<NOT:$<BOOL:${THEROCK_BUILD_TESTING}>>"
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipDNN
      therock-flatbuffers
      therock-nlohmann-json
    RUNTIME_DEPS
      miopenprovider
    TEST_SUBPROJECTS
  )
  therock_cmake_subproject_glob_c_sources(hipDNN_samples
  SUBDIRS
    .
  )
  therock_cmake_subproject_activate(hipDNN_samples)

  therock_provide_artifact(hipdnn-samples
    DESCRIPTOR artifact-hipdnn-samples.toml
    COMPONENTS
      test
    SUBPROJECT_DEPS
      hipDNN_samples
  )
endif(THEROCK_ENABLE_HIPDNN_SAMPLES)
