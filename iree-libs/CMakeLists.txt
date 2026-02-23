##############################################################################
# IREE Compiler - builds libIREECompiler.so
##############################################################################
if(THEROCK_ENABLE_IREE_COMPILER)
  therock_cmake_subproject_declare(iree-compiler
    NO_MERGE_COMPILE_COMMANDS
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/iree-compiler"
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree"
    CMAKE_ARGS
      # Build the compiler
      -DIREE_BUILD_COMPILER=ON
      -DIREE_TARGET_BACKEND_DEFAULTS=OFF
      -DIREE_TARGET_BACKEND_ROCM=ON
      -DIREE_TARGET_BACKEND_LLVM_CPU=ON
      # Only enable input dialects Fusilli generates
      -DIREE_INPUT_TORCH=ON
      -DIREE_INPUT_STABLEHLO=OFF
      # Use HIP headers from TheRock's build
      -DIREE_USE_SYSTEM_DEPS=ON
      # In this targeted build, several submodules aren't needed
      -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
      # Disable tests, samples, and bindings
      -DIREE_BUILD_TESTS=OFF
      -DIREE_BUILD_SAMPLES=OFF
      -DIREE_BUILD_BINDINGS_TFLITE=OFF
      -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF
      # Disable runtime, it's built into fusilli, we only need the compiler
      -DIREE_HAL_DRIVER_DEFAULTS=OFF
    BUILD_DEPS
      hip-clr  # Provides HIP headers for IREE_USE_SYSTEM_DEPS
  )
  therock_cmake_subproject_activate(iree-compiler)

  therock_provide_artifact(iree-compiler
    TARGET_NEUTRAL
    DESCRIPTOR artifact-iree-compiler.toml
    COMPONENTS
      lib
      dbg
    SUBPROJECT_DEPS
      iree-compiler
  )
endif()

if(THEROCK_ENABLE_FUSILLI_PLUGIN)
  ##############################################################################
  # fusilli library
  ##############################################################################
  therock_cmake_subproject_declare(fusilli
    USE_DIST_AMDGPU_TARGETS
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilli"
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli"
    RUNTIME_DEPS
      hip-clr
      rocminfo  # Provides rocm_agent_enumerator
    CMAKE_ARGS
      -DFUSILLI_BUILD_TESTS=OFF
      -DFUSILLI_BUILD_BENCHMARKS=OFF
      -DFUSILLI_SYSTEMS_AMDGPU=ON
      -DFUSILLI_CODE_COVERAGE=OFF
      -DFUSILLI_ENABLE_LOGGING=OFF
      -DFUSILLI_ENABLE_CLANG_TIDY=OFF
      -DIREE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/iree
      -DHIP_PLATFORM=amd
      -DIREE_USE_SYSTEM_DEPS=ON
  )
  therock_cmake_subproject_glob_c_sources(fusilli
    SUBDIRS
      .
  )
  therock_cmake_subproject_provide_package(fusilli Fusilli lib/cmake/fusilli)
  therock_cmake_subproject_activate(fusilli)

  ##############################################################################
  # fusilli hipDNN plugin
  ##############################################################################
  therock_cmake_subproject_declare(fusilli-plugin
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli/plugins/hipdnn-plugin"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilli-plugin"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DFUSILLI_PLUGIN_SKIP_TESTS=$<NOT:$<BOOL:${THEROCK_BUILD_TESTING}>>
      -DIREE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/iree
      -DIREE_USE_SYSTEM_DEPS=ON
      -DHIP_PLATFORM=amd
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      fusilli
      hipDNN
      therock-flatbuffers
      therock-fmt
      therock-googletest
      therock-nlohmann-json
      therock-spdlog
    RUNTIME_DEPS
      hip-clr
      hipDNN
      iree-compiler  # Provides libIREECompiler.so for CAPI compilation
      rocminfo  # Provides rocm_agent_enumerator
  )
  therock_cmake_subproject_glob_c_sources(fusilli-plugin
    SUBDIRS
      .
  )
  therock_cmake_subproject_activate(fusilli-plugin)

  therock_provide_artifact(fusilli-plugin
    DESCRIPTOR artifact-fusilli-plugin.toml
    COMPONENTS
      dbg
      lib
      test
    SUBPROJECT_DEPS
      fusilli-plugin
  )
endif(THEROCK_ENABLE_FUSILLI_PLUGIN)
