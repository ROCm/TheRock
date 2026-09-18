# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(DEBUG_TOOLS_DIR "${THEROCK_SOURCE_DIR}/debug-tools")

# If set, propagate DEBUG_TOOLS_BUILD_TYPE to all debug-tools components.
if(DEBUG_TOOLS_BUILD_TYPE)
  message(STATUS "Setting all debug-tools components to CMAKE_BUILD_TYPE=${DEBUG_TOOLS_BUILD_TYPE}")
  if(NOT amd-dbgapi_BUILD_TYPE)
    set(amd-dbgapi_BUILD_TYPE "${DEBUG_TOOLS_BUILD_TYPE}")
  endif()
  if(NOT rocr-debug-agent_BUILD_TYPE)
    set(rocr-debug-agent_BUILD_TYPE "${DEBUG_TOOLS_BUILD_TYPE}")
  endif()
  if(NOT rocgdb_BUILD_TYPE)
    set(rocgdb_BUILD_TYPE "${DEBUG_TOOLS_BUILD_TYPE}")
  endif()
endif()

# Allow custom debug flags for debug-tools components.
# Example: -DDEBUG_TOOLS_C_FLAGS_DEBUG="-O0 -g3"
#          -DDEBUG_TOOLS_CXX_FLAGS_DEBUG="-O0 -g3"
if(DEBUG_TOOLS_C_FLAGS_DEBUG OR DEBUG_TOOLS_CXX_FLAGS_DEBUG)
  set(_debug_tools_cmake_args)
  if(DEBUG_TOOLS_C_FLAGS_DEBUG)
    message(STATUS "Setting debug-tools CMAKE_C_FLAGS_DEBUG=${DEBUG_TOOLS_C_FLAGS_DEBUG}")
    list(APPEND _debug_tools_cmake_args "-DCMAKE_C_FLAGS_DEBUG=${DEBUG_TOOLS_C_FLAGS_DEBUG}")
  endif()
  if(DEBUG_TOOLS_CXX_FLAGS_DEBUG)
    message(STATUS "Setting debug-tools CMAKE_CXX_FLAGS_DEBUG=${DEBUG_TOOLS_CXX_FLAGS_DEBUG}")
    list(APPEND _debug_tools_cmake_args "-DCMAKE_CXX_FLAGS_DEBUG=${DEBUG_TOOLS_CXX_FLAGS_DEBUG}")
  endif()

  # Propagate to all components CMAKE_ARGS.
  if(NOT amd-dbgapi_CMAKE_ARGS)
    set(amd-dbgapi_CMAKE_ARGS "${_debug_tools_cmake_args}")
  else()
    list(APPEND amd-dbgapi_CMAKE_ARGS ${_debug_tools_cmake_args})
  endif()

  if(NOT rocr-debug-agent_CMAKE_ARGS)
    set(rocr-debug-agent_CMAKE_ARGS "${_debug_tools_cmake_args}")
  else()
    list(APPEND rocr-debug-agent_CMAKE_ARGS ${_debug_tools_cmake_args})
  endif()

  if(NOT rocgdb_CMAKE_ARGS)
    set(rocgdb_CMAKE_ARGS "${_debug_tools_cmake_args}")
  else()
    list(APPEND rocgdb_CMAKE_ARGS ${_debug_tools_cmake_args})
  endif()
endif()

if(THEROCK_ENABLE_AMD_DBGAPI)
  ##############################################################################
  # amd-dbgapi
  #
  # A library that provides support for a debugger and other tools to perform
  # low-level control of the running code and inspection of the running state
  # of AMD GPU's.
  ##############################################################################

  if(THEROCK_USE_EXTERNAL_AMD_DBGAPI AND THEROCK_AMD_DBGAPI_SOURCE_DIR)
    message(WARNING "Configuring amd-dbgapi with modified source directory: ${THEROCK_AMD_DBGAPI_SOURCE_DIR}")
  endif()

  # amd-dbgapi is a shared library on Windows too: rocgdb is built by the
  # ucrt64 gcc, which cannot link an MSVC-built static amd-dbgapi, whose C++
  # pulls in the MSVC C++ runtime.  The DLL's name carries its version
  # (rocm-dbgapi-0-84.dll and so on), so a mismatched rocgdb fails to start.

  # Only build and link against GoogleTest when tests are requested, so
  # the dependency isn't pulled in for non-testing configurations.
  set(EXTRA_AMD_DBGAPI_BUILD_DEPS)
  if(THEROCK_BUILD_TESTING)
    list(APPEND EXTRA_AMD_DBGAPI_BUILD_DEPS therock-googletest)
  endif()

  therock_cmake_subproject_declare(amd-dbgapi
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_AMD_DBGAPI_SOURCE_DIR}"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/amd-dbgapi"
    COMPILER_TOOLCHAIN
      amd-llvm
    CMAKE_ARGS
      -DTHE_ROCK_BUILD=ON
      "-DENABLE_TESTS=${THEROCK_BUILD_TESTING}"
    BUILD_DEPS
      ${THEROCK_BUNDLED_LIBBACKTRACE}
      ${EXTRA_AMD_DBGAPI_BUILD_DEPS}
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
  if(NOT WIN32)
    therock_test_validate_shared_lib(PATH amd-dbgapi/dist/lib
      LIB_NAMES
        librocm-dbgapi.so
    )
  endif()

  # The artifacts provided by amd-dbgapi.
  therock_provide_artifact(amd-dbgapi
    TARGET_NEUTRAL
    DESCRIPTOR artifact-amd-dbgapi.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      test
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

  if(THEROCK_USE_EXTERNAL_ROCR_DEBUG_AGENT AND THEROCK_ROCR_DEBUG_AGENT_SOURCE_DIR)
    message(WARNING "Configuring rocr-debug-agent with modified source directory: ${THEROCK_ROCR_DEBUG_AGENT_SOURCE_DIR}")
  endif()

  # Make sure to pass -DENABLE_TESTS=OFF so we don't build
  # rocr-debug-agent-tests alongside the library. The tests need to be built
  # for each gfx level that is requested by the build.
  therock_cmake_subproject_declare(rocr-debug-agent
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCR_DEBUG_AGENT_SOURCE_DIR}"
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
  # whenever -DTHEROCK_BUILD_TESTING=ON. If multiple GPU targets are specified,
  # fat binaries are built containing support for all of those GPU targets.
  ##############################################################################

  if(THEROCK_USE_EXTERNAL_ROCR_DEBUG_AGENT AND THEROCK_ROCR_DEBUG_AGENT_SOURCE_DIR)
    message(WARNING "Configuring rocr-debug-agent-tests with modified source directory: ${THEROCK_ROCR_DEBUG_AGENT_SOURCE_DIR}/test")
  endif()

  if(THEROCK_BUILD_TESTING)
    therock_cmake_subproject_declare(rocr-debug-agent-tests
      USE_TEST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCR_DEBUG_AGENT_SOURCE_DIR}/test"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocr-debug-agent-tests"
      INSTALL_RPATH_EXECUTABLE_DIR
        "tests/rocm-debug-agent"
      CMAKE_ARGS
        "-DHIP_PLATFORM=amd"
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        therock-googletest
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
      TARGET_NEUTRAL
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
  # The AMD source-level debugger, based on the GNU Debugger (GDB).
  ##############################################################################
  if(WIN32)
    # rocgdb links the bundled sysdeps, and Windows has no system ones to fall
    # back on.  Checked here, when the super-project configures, instead of in
    # rocgdb's own configure, which runs only once the build reaches rocgdb.
    if(NOT THEROCK_BUNDLE_SYSDEPS)
      message(FATAL_ERROR
        "rocgdb on Windows requires THEROCK_BUNDLE_SYSDEPS=ON: it links the "
        "bundled sysdeps, and there are no system ones to fall back on.")
    endif()

    # Windows builds the no-python variant only, for now.  Every variant
    # installs as bin/rocgdb.exe there, with no chooser script to pick between
    # them, so a python variant would overwrite it.  Do not look for python at
    # all, instead of depending on whether the host's python reports a shared
    # libpython.
    if(THEROCK_SHARED_PYTHON_EXECUTABLES)
      message(STATUS "rocgdb is built without Python on Windows, ignoring "
        "THEROCK_SHARED_PYTHON_EXECUTABLES")
    endif()
    message(STATUS "Building rocgdb without Python support")
  else()
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
  endif()

  if(THEROCK_USE_EXTERNAL_ROCGDB AND THEROCK_ROCGDB_SOURCE_DIR)
    message(WARNING "Configuring ROCgdb with modified source directory: ${THEROCK_ROCGDB_SOURCE_DIR}")
  endif()

  # Default to building ROCm's rocgdb branch (with NOTICES.txt, roccoremerge, etc.).
  # For upstream master builds, which currently lack AMD-specific files:
  #   - NOTICES.txt file
  #   - roccoremerge program
  #   - roccoremerge man page
  # set: -DTHEROCK_ROCGDB_UPSTREAM_BUILD=ON
  # Note: upstream master will eventually include roccoremerge.
  option(THEROCK_ROCGDB_UPSTREAM_BUILD
    "Build rocgdb against upstream GDB master (skips AMD-specific files like NOTICES.txt, roccoremerge)"
    OFF)

  # When ON, download the test_rocgdb.py CI wrapper script, its
  # rocjitsu_emulator.py module and rocgdb_emulate.py from ROCm/ROCgdb
  # amd-staging if any is missing from the rocgdb source tree, and install
  # them. Required by TheRock CI when building a rocgdb source tree that is
  # not part of the branches exercised by CI.
  # Off by default to avoid a network fetch at configure time.
  option(THEROCK_ROCGDB_DOWNLOAD_CI_SCRIPT
    "Download test_rocgdb.py, rocjitsu_emulator.py and rocgdb_emulate.py from ROCm/ROCgdb amd-staging if missing in rocgdb source tree"
    OFF)

  # Which compiler builds rocgdb differs by host.  On Linux it is the
  # TheRock-built amd-llvm.  On Windows gdb has never been ported to the
  # windows-msvc target, so it needs a compiler targeting mingw, and TheRock's
  # amd-llvm targets the MSVC ABI.  host-gnu names the MSYS2 ucrt64 gcc instead.
  # Declaring it also gets the gcc-form -L for the bundled sysdeps, and leaves
  # the compile flags to the recipe.
  #
  # Naming amd-llvm on Windows would declare that its clang compiles rocgdb,
  # which it does not.  That choice does not change what gets built first:
  # amd-comgr below is part of amd-llvm, so it is built ahead of rocgdb either
  # way -- as an artifact rocgdb links against, not as the compiler that builds
  # it.
  set(_rocgdb_toolchain amd-llvm)
  if(WIN32)
    set(_rocgdb_toolchain host-gnu)
  endif()
  therock_autotools_subproject_args(rocgdb _autotools_args)

  therock_cmake_subproject_declare(rocgdb
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "rocgdb"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocgdb"
    ${_autotools_args}
    COMPILER_TOOLCHAIN ${_rocgdb_toolchain}
    CMAKE_ARGS
      "-DPATCHELF=${PATCHELF}"
      "-DBUILD_TESTING=${THEROCK_BUILD_TESTING}"
      "-DROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
      "-DSHARED_PYTHON_EXECUTABLES=${_rocgdb_python_executables_escaped}"
      "-DTHEROCK_BUNDLE_SYSDEPS=${THEROCK_BUNDLE_SYSDEPS}"
      "-DROCGDB_SOURCE_DIR=${THEROCK_ROCGDB_SOURCE_DIR}"
      "-DROCGDB_UPSTREAM_BUILD=${THEROCK_ROCGDB_UPSTREAM_BUILD}"
      "-DROCGDB_DOWNLOAD_CI_SCRIPT=${THEROCK_ROCGDB_DOWNLOAD_CI_SCRIPT}"
    RUNTIME_DEPS
      amd-comgr
      amd-dbgapi
      ${THEROCK_BUNDLED_BZIP2}
      ${THEROCK_BUNDLED_GMP}
      ${THEROCK_BUNDLED_MPFR}
      ${THEROCK_BUNDLED_EXPAT}
      ${THEROCK_BUNDLED_NCURSES}
      ${THEROCK_BUNDLED_LIBLZMA}
      ${THEROCK_BUNDLED_LIBICONV}
      ${THEROCK_BUNDLED_ZLIB}
      ${THEROCK_BUNDLED_ZSTD}
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
      ${THEROCK_BUNDLED_BZIP2}
      ${THEROCK_BUNDLED_EXPAT}
      ${THEROCK_BUNDLED_GMP}
      ${THEROCK_BUNDLED_MPFR}
      ${THEROCK_BUNDLED_NCURSES}
      ${THEROCK_BUNDLED_LIBLZMA}
      ${THEROCK_BUNDLED_LIBICONV}
      ${THEROCK_BUNDLED_ZLIB}
      ${THEROCK_BUNDLED_ZSTD}
      rocgdb
  )
endif()
