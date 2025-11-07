set(DEBUG_TOOLS_DIR "${THEROCK_SOURCE_DIR}/debug-tools")

if(THEROCK_ENABLE_AMD_DBGAPI)
  ##############################################################################
  # amd-dbgapi
  #
  # A library that provides support for a debugger and other tools to perform
  # low-level control of the running code and inspection of the running state
  # of AMD GPU's.
  ##############################################################################

  therock_cmake_subproject_declare(amd-dbgapi
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "amd-dbgapi"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/amd-dbgapi"
    COMPILER_TOOLCHAIN
      amd-llvm
    CMAKE_ARGS
      -DTHE_ROCK_BUILD=ON
    BUILD_DEPS
      ROCR-Runtime
      therock-libbacktrace
    RUNTIME_DEPS
      amd-comgr
    INTERFACE_LINK_DIRS
      "lib"
    INTERFACE_INSTALL_RPATH_DIRS
      "lib"
    INTERFACE_PKG_CONFIG_DIRS
      "share/pkgconfig"
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
endif()
