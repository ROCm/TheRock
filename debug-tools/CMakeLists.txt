set(DEBUG_TOOLS_DIR "${THEROCK_SOURCE_DIR}/debug-tools")

if(THEROCK_ENABLE_DEBUG_TOOLS_ROCDBGAPI)
  ##############################################################################
  # amd-dbgapi
  #
  # A library that provides support for a debugger and other tools to perform
  # low-level control of the running code and inspection of the running state
  # of AMD GPU's
  ##############################################################################

  therock_cmake_subproject_declare(amd-dbgapi
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "amd-dbgapi"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/amd-dbgapi"
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN
      amd-llvm
    CMAKE_ARGS
      -DREQUIRE_THREADS=1
    BUILD_DEPS
      ROCR-Runtime
      therock-libbacktrace
    RUNTIME_DEPS
      amd-comgr
    INTERFACE_LINK_DIRS
      "lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib"
  )

  therock_cmake_subproject_glob_c_sources(amd-dbgapi SUBDIRS
    include
    src
  )

  therock_cmake_subproject_provide_package(amd-dbgapi amd-dbgapi
    lib/cmake/amd-dbgapi
  )

  therock_cmake_subproject_activate(amd-dbgapi)

  # Check that we built the one library we're supposed to build.
  therock_test_validate_shared_lib(PATH amd-dbgapi/dist/lib
    LIB_NAMES
      librocm-dbgapi.so
  )

  # The artifacts provided by amd-dbgapi.
  therock_provide_artifact(amd-dbgapi
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-dbgapi.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
    SUBPROJECT_DEPS
      amd-comgr
      amd-dbgapi
  )

  # To speed up the transition of amd-dbgapi to TheRock, we keep a separate
  # cmake file that we copy to the sources when the project is setup.
  #
  # This step is temporary and will eventually disappear in favor of having
  # the right changes in the project's build system.
  message(STATUS "[amd-dbgapi] Copying amd-dbgapi.cmake to the source directory")
  execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy ${DEBUG_TOOLS_DIR}/amd-dbgapi.cmake ${DEBUG_TOOLS_DIR}/amd-dbgapi/CMakeLists.txt
      OUTPUT_VARIABLE _output
      ERROR_VARIABLE _error
      RESULT_VARIABLE _result
  )
endif()

if(THEROCK_ENABLE_DEBUG_TOOLS_ROCR_DEBUG_AGENT)
  ##############################################################################
  # rocr-debug-agent
  #
  # A library that can be loaded by the ROCm software runtime to print the
  # state of wavefronts.
  ##############################################################################

  # Make sure to pass -DENABLE_TESTS=OFF so we don't build
  # rocr-debug-agent-tests alongside the library. The tests need to be built
  # for each gfx level that is requested by the build.
  therock_cmake_subproject_declare(rocr-debug-agent
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "rocr-debug-agent"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocr-debug-agent"
    BACKGROUND_BUILD
    CMAKE_ARGS
      "-DENABLE_TESTS=OFF"
    COMPILER_TOOLCHAIN
      amd-llvm
    RUNTIME_DEPS
      amd-dbgapi
      ROCR-Runtime
      ${THEROCK_BUNDLED_ELFUTILS}
    INTERFACE_LINK_DIRS
      "lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib"
  )

  therock_cmake_subproject_glob_c_sources(rocr-debug-agent SUBDIRS
    src
  )

  therock_cmake_subproject_provide_package(rocr-debug-agent rocr-debug-agent
    lib/cmake/rocr-debug-agent)
  therock_cmake_subproject_activate(rocr-debug-agent)

  # Check that we built the one library we're supposed to build.
  therock_test_validate_shared_lib(PATH rocr-debug-agent/dist/lib
    LIB_NAMES
      librocm-debug-agent.so
  )

  # The artifacts provided by rocr-debug-agent.
  therock_provide_artifact(rocr-debug-agent
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rocr-debug-agent.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
    SUBPROJECT_DEPS
      ${THEROCK_BUNDLED_ELFUTILS}
      amd-dbgapi
      rocr-debug-agent
      ROCR-Runtime
  )

  # To speed up the transition of rocr-debug-agent to TheRock, we keep a
  # separate cmake file that we copy to the sources when the project is
  # setup.
  #
  # This step is temporary and will eventually disappear in favor of having
  # the right changes in the project's build system.
  message(STATUS "[rocr-debug-agent] Copying rocr-debug-agent.cmake to the source directory")
  execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy ${DEBUG_TOOLS_DIR}/rocr-debug-agent.cmake ${DEBUG_TOOLS_DIR}/rocr-debug-agent/CMakeLists.txt
      OUTPUT_VARIABLE _output
      ERROR_VARIABLE _error
      RESULT_VARIABLE _result
  )

  message(STATUS "[rocr-debug-agent-tests] Copying rocr-debug-agent-tests.cmake to the source directory")
  execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy ${DEBUG_TOOLS_DIR}/rocr-debug-agent-tests.cmake ${DEBUG_TOOLS_DIR}/rocr-debug-agent/test/CMakeLists.txt
      OUTPUT_VARIABLE _output
      ERROR_VARIABLE _error
      RESULT_VARIABLE _result
  )


  ##############################################################################
  # rocr-debug-agent-tests
  #
  # Validation tests for the rocr-debug-agent component. This is only built
  # whenever -DBUILD_TESTING=ON and we may have a build for each gfx level
  # that is requested.
  ##############################################################################

  if(BUILD_TESTING)
    therock_cmake_subproject_declare(rocr-debug-agent-tests
      EXTERNAL_SOURCE_DIR "rocr-debug-agent/test"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocr-debug-agent-tests"
      BACKGROUND_BUILD
      CMAKE_ARGS
        "-DHIP_PLATFORM=amd"
      COMPILER_TOOLCHAIN
        amd-hip
      RUNTIME_DEPS
        amd-dbgapi
        hip-clr
        rocr-debug-agent
        ROCR-Runtime
        ${THEROCK_BUNDLED_ELFUTILS}
    )

    therock_cmake_subproject_glob_c_sources(rocr-debug-agent-tests SUBDIRS
      .
    )

    therock_cmake_subproject_provide_package(rocr-debug-agent-tests rocr-debug-agent-tests
      lib/cmake/rocr-debug-agent-tests)
    therock_cmake_subproject_activate(rocr-debug-agent-tests)

    # The artifacts provided by rocr-debug-agent-tests.
    therock_provide_artifact(rocr-debug-agent-tests
      DESCRIPTOR artifact-rocr-debug-agent-tests.toml
      COMPONENTS
        doc
        test
      SUBPROJECT_DEPS
        ${THEROCK_BUNDLED_ELFUTILS}
        amd-dbgapi
        hip-clr
        rocr-debug-agent
        rocr-debug-agent-tests
        ROCR-Runtime
    )
  endif()
endif()
