if(THEROCK_ENABLE_FUSILLIPROVIDER)
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
  # fusilli hipDNN provider
  ##############################################################################
  therock_cmake_subproject_declare(fusilliprovider
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli/plugins/hipdnn-plugin"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilliprovider"
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
      rocminfo  # Provides rocm_agent_enumerator
  )
  therock_cmake_subproject_glob_c_sources(fusilliprovider
    SUBDIRS
      .
  )
  therock_cmake_subproject_activate(fusilliprovider)

  therock_provide_artifact(fusilliprovider
    DESCRIPTOR artifact-fusilliprovider.toml
    COMPONENTS
      dbg
      lib
      test
    SUBPROJECT_DEPS
      fusilliprovider
  )
endif(THEROCK_ENABLE_FUSILLIPROVIDER)
