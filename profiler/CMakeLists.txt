if(THEROCK_ENABLE_PROFILER_SDK)

  ##############################################################################
  # aqlprofile
  ##############################################################################
  therock_cmake_subproject_declare(aqlprofile
    EXTERNAL_SOURCE_DIR "aqlprofile"
    BACKGROUND_BUILD
    INTERFACE_LINK_DIRS
      lib  # So that dependents can find the aqlprofile library via find_library
    RUNTIME_DEPS
      amd-llvm
      ROCR-Runtime
  )
  therock_cmake_subproject_activate(aqlprofile)

  ##############################################################################
  # rocprofiler-sdk
  ##############################################################################
  therock_cmake_subproject_declare(rocprofiler-sdk
    EXTERNAL_SOURCE_DIR "rocprofiler-sdk"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
    RUNTIME_DEPS
      aqlprofile
      hip-clr
      rocprofiler-register
      ${THEROCK_BUNDLED_ELFUTILS}
      ${THEROCK_BUNDLED_LIBDRM}
      ${THEROCK_BUNDLED_SQLITE3}
  )
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk lib/cmake/rocprofiler-sdk)
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk-roctx lib/cmake/rocprofiler-sdk-roctx)
  therock_cmake_subproject_activate(rocprofiler-sdk)

  ##############################################################################
  # roctracer-compat
  ##############################################################################

  therock_cmake_subproject_declare(roctracer-compat
    EXTERNAL_SOURCE_DIR "roctracer-compat"
    BACKGROUND_BUILD
    RUNTIME_DEPS
      rocprofiler-sdk
  )
  therock_cmake_subproject_glob_c_sources(roctracer-compat
    SUBDIRS .
  )
  therock_cmake_subproject_provide_package(roctracer-compat roctracer-compat lib/cmake/roctracer-compat)
  therock_cmake_subproject_activate(roctracer-compat)

  therock_provide_artifact(rocprofiler-sdk
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rocprofiler-sdk.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS
      aqlprofile
      rocprofiler-sdk
      roctracer-compat
  )

endif(THEROCK_ENABLE_PROFILER_SDK)
