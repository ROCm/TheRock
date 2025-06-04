if(THEROCK_ENABLE_ROCPROFV3)

  ##############################################################################
  # aqlprofile
  ##############################################################################
  therock_cmake_subproject_declare(aqlprofile
    # TODO: Tests do not build properly for explicit targets: Switch to
    # USE_DIST_AMDGPU_TARGETS once fixed.
    # See: https://github.com/ROCm/aqlprofile/issues/2
    DISABLE_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "aqlprofile"
    BACKGROUND_BUILD
    INTERFACE_LINK_DIRS
      lib  # So that dependents can find the aqlprofile library via find_library
    RUNTIME_DEPS
      amd-llvm
      ROCR-Runtime
  )
  therock_cmake_subproject_glob_c_sources(aqlprofile
    SUBDIRS .
  )
  therock_cmake_subproject_activate(aqlprofile)

  ##############################################################################
  # rocprofiler-sdk
  ##############################################################################
  therock_cmake_subproject_declare(rocprofiler-sdk
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "rocprofiler-sdk"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    RUNTIME_DEPS
      aqlprofile
      hip-clr
      rocprofiler-register
      ${THEROCK_BUNDLED_ELFUTILS}
      ${THEROCK_BUNDLED_LIBDRM}
      ${THEROCK_BUNDLED_SQLITE3}
  )
  therock_cmake_subproject_glob_c_sources(rocprofiler-sdk
    SUBDIRS .
  )
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk lib/cmake/rocprofiler-sdk)
  therock_cmake_subproject_provide_package(rocprofiler-sdk rocprofiler-sdk-roctx lib/cmake/rocprofiler-sdk-roctx)
  therock_cmake_subproject_activate(rocprofiler-sdk)

  ##############################################################################
  # roctracer
  # This is a very old, deprecated library with a number of quirks. Long term,
  # everything is supposed to migrate to rocprofiler-sdk, which has runtime
  # support to intercept any old roctracer clients when tracing is active.
  # To prepare for removal, anything that depends on roctracer should also
  # depend on rocprofiler-sdk.
  ##############################################################################

  therock_cmake_subproject_declare(roctracer
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "roctracer"
    BACKGROUND_BUILD
    COMPILER_TOOLCHAIN
      # Must build with the HIP compiler.
      amd-hip
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
    INTERFACE_INCLUDE_DIRS
      # All old clients of roctx64 expect to just be able to find its include
      # with no further qualification. Bad design, but also deprecated, so meh.
      include
    INTERFACE_LINK_DIRS
      # So that dependents can find the roctx library via find_library()
      lib
    RUNTIME_DEPS
      hip-clr
      ROCR-Runtime
  )
  therock_cmake_subproject_glob_c_sources(roctracer
    SUBDIRS .
  )
  therock_cmake_subproject_activate(roctracer)

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
      roctracer
  )

endif(THEROCK_ENABLE_ROCPROFV3)
