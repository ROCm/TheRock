if(THEROCK_ENABLE_DEBUG_TOOLS)

  ##############################################################################
  # amd-dbgapi
  #
  # A library that provides support for a debugger and other tools to perform
  # low-level control of the running code and inspection of the running state
  # of AMD GPU's
  ##############################################################################
  if(NOT WIN32) # Disabled on Windows for now.

    therock_cmake_subproject_declare(amd-dbgapi
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocdbgapi"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/amd-dbgapi"
      BACKGROUND_BUILD
      COMPILER_TOOLCHAIN
        amd-llvm
      BUILD_DEPS
        ROCR-Runtime
      RUNTIME_DEPS
        amd-comgr
      INTERFACE_LINK_DIRS
        "lib"
      INTERFACE_INSTALL_RPATH_DIRS
        "lib"
    )

    therock_cmake_subproject_glob_c_sources(amd-dbgapi SUBDIRS
      doc
      include
      src
    )

    therock_cmake_subproject_provide_package(amd-dbgapi amd-dbgapi
      lib/cmake/amd-dbgapi)
    therock_cmake_subproject_activate(amd-dbgapi)

    # Check that we built the one library we're supposed to build.
    therock_test_validate_shared_lib(PATH amd-dbgapi/dist/lib
      LIB_NAMES
        librocm-dbgapi.so
    )
  endif()

  ##############################################################################
  # rocr-debug-agent
  #
  # A library that can be loaded by the ROCm software runtime to print the
  # state of wavefronts.
  ##############################################################################
  if(NOT WIN32) # Disabled on Windows for now.

    therock_cmake_subproject_declare(rocm-debug-agent
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocr-debug-agent"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocm-debug-agent"
      BACKGROUND_BUILD
      CMAKE_ARGS
        "-DHIP_PLATFORM=amd"
      COMPILER_TOOLCHAIN
        amd-hip
      BUILD_DEPS
        hip-clr
      RUNTIME_DEPS
        amd-dbgapi
        ROCR-Runtime
        ${THEROCK_BUNDLED_ELFUTILS}
    )

    # We may want to adjust this to optimize rebuilds.
    therock_cmake_subproject_glob_c_sources(rocm-debug-agent SUBDIRS .)

    therock_cmake_subproject_provide_package(rocm-debug-agent rocm-debug-agent
      lib/cmake/rocr-debug-agent)
    therock_cmake_subproject_activate(rocm-debug-agent)

    # Check that we built the one library we're supposed to build.
    therock_test_validate_shared_lib(PATH rocm-debug-agent/dist/lib
      LIB_NAMES
        librocm-debug-agent.so
    )
  endif()

  ##############################################################################
  # ROCgdb
  #
  # The AMD source-level debugger for Linux, based on the GNU Debugger (GDB).
  ##############################################################################
  if(NOT WIN32) # Disabled on Windows for now.

    therock_cmake_subproject_declare(rocgdb
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocgdb"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rocgdb"
      BACKGROUND_BUILD
      COMPILER_TOOLCHAIN
        amd-llvm
      CMAKE_ARGS
        -DROCM_PATH=
        -DROCM_DIR=
      RUNTIME_DEPS
        amd-comgr
        amd-dbgapi
    )

    # We may want to adjust this to optimize rebuilds.
    therock_cmake_subproject_glob_c_sources(rocgdb SUBDIRS
      bfd
      gdb
      gdbsupport
      includes
      binutils
    )
    therock_cmake_subproject_provide_package(rocgdb rocgdb lib/cmake/rocgdb)
    therock_cmake_subproject_activate(rocgdb)
  endif()

endif(THEROCK_ENABLE_DEBUG_TOOLS)
