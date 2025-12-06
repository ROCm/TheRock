if(THEROCK_ENABLE_FUSILLI_PLUGIN)
  ##############################################################################
  # fusilli library
  ##############################################################################

  therock_cmake_subproject_declare(therock-fusilli
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
  therock_cmake_subproject_provide_package(therock-fusilli Fusilli lib/cmake/fusilli)
  therock_cmake_subproject_activate(therock-fusilli)

  add_dependencies(therock-third-party therock-fusilli)

  ##############################################################################
  # fusilli hipDNN plugin
  ##############################################################################

  therock_cmake_subproject_declare(fusilli_plugin
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fusilli/plugins/hipdnn-plugin"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fusilli_plugin"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DFUSILLI_PLUGIN_SKIP_TESTS=$<NOT:$<BOOL:${THEROCK_BUILD_TESTING}>>
      -DIREE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/iree
      -DIREE_USE_SYSTEM_DEPS=ON
      -DHIP_PLATFORM=amd
    CMAKE_INCLUDES
      therock_explicit_finders.cmake # TODO: is this necessary? I have no idea what this does.
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipDNN
      therock-googletest
      therock-flatbuffers
      therock-nlohmann-json
      therock-fmt
      therock-spdlog
      therock-fusilli
    RUNTIME_DEPS
      hip-clr
      hipDNN
      rocminfo  # Provides rocm_agent_enumerator
  )
  therock_cmake_subproject_glob_c_sources(fusilli_plugin
  SUBDIRS
    .
  )
  therock_cmake_subproject_activate(fusilli_plugin)

  therock_provide_artifact(fusilli-plugin
    DESCRIPTOR artifact-fusilli-plugin.toml
    COMPONENTS
      dbg
      lib
      test
    SUBPROJECT_DEPS
      fusilli_plugin
  )
endif(THEROCK_ENABLE_FUSILLI_PLUGIN)
