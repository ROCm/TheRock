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

if(THEROCK_ENABLE_ROCR_DEBUG_AGENT)
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
endif()

if(THEROCK_ENABLE_ROCR_DEBUG_AGENT_TESTS)
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

if(THEROCK_ENABLE_ROCGDB)
  ##############################################################################
  # rocgdb
  #
  # The AMD source-level debugger for Linux, based on the GNU Debugger (GDB).
  ##############################################################################
  therock_detect_shared_python_executables(_rocgdb_python_executables)
  if(_rocgdb_python_executables)
    message(STATUS "Building rocgdb against the following libpython versions: ${_rocgdb_python_executables}")
    # Escape semicolons so the list survives a call to
    # therock_cmake_subproject_declare and also add_custom_command
    # expansion in CMAKE_ARGS.
    string(REPLACE ";" "\\\$<SEMICOLON>" _rocgdb_python_executables_escaped "${_rocgdb_python_executables}")
  else()
    message(WARNING "No Python with shared libpython found. rocgdb will be built without Python support.")
  endif()

  therock_cmake_subproject_declare(rocgdb
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "rocgdb"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocgdb"
    CMAKE_ARGS
      "-DPATCHELF=${PATCHELF}"
      "-DBUILD_TESTING=${BUILD_TESTING}"
      "-DROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
      "-DSHARED_PYTHON_EXECUTABLES=${_rocgdb_python_executables_escaped}"
    COMPILER_TOOLCHAIN
      amd-llvm
    RUNTIME_DEPS
      amd-comgr
      amd-dbgapi
      therock-bzip2
      therock-gmp
      therock-mpfr
      therock-expat
      therock-ncurses
      therock-liblzma
      therock-zlib
      therock-zstd
    INTERFACE_PROGRAM_DIRS
      "bin"
    INTERFACE_LINK_DIRS
      "lib"
  )

  # We may want to adjust this to optimize rebuilds.
  therock_cmake_subproject_glob_c_sources(rocgdb SUBDIRS
    bfd
    binutils
    gdb
    gdbsupport
    includes
  )
  therock_cmake_subproject_provide_package(rocgdb rocgdb lib/cmake/rocgdb)
  therock_cmake_subproject_activate(rocgdb)

  # The artifacts provided by rocgdb.
  therock_provide_artifact(rocgdb
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rocgdb.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      amd-comgr
      amd-dbgapi
      therock-bzip2
      therock-expat
      therock-gmp
      therock-mpfr
      therock-ncurses
      therock-liblzma
      therock-zlib
      therock-zstd
      rocgdb
  )
endif()
