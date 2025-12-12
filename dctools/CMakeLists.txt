# Data Center Tools

if(THEROCK_ENABLE_RDC)
  if(WIN32)
    message(FATAL_ERROR "RDC requires Linux (Windows not supported)")
  endif()

  # Get gRPC stage directory for RDC's GRPC_ROOT requirement
  get_target_property(_grpc_stage therock-grpc THEROCK_STAGE_DIR)

  therock_cmake_subproject_declare(rdc
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rdc"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rdc"
    BACKGROUND_BUILD
    USE_DIST_AMDGPU_TARGETS

    CMAKE_ARGS
      -DBUILD_PROFILER=ON
      -DBUILD_STANDALONE=ON
      -DBUILD_RUNTIME=ON
      -DBUILD_RVS=OFF
      -DBUILD_TESTS=${THEROCK_BUILD_TESTING}
      -DHIP_PLATFORM=amd
      -DCMAKE_CXX_STANDARD=17
      -DGRPC_ROOT=${_grpc_stage}

    BUILD_DEPS
      therock-grpc

    RUNTIME_DEPS
      ROCR-Runtime
      amdsmi
      rocprofiler-sdk
      ${THEROCK_BUNDLED_LIBCAP}
      ${THEROCK_BUNDLED_ZLIB}

    INSTALL_DESTINATION
      portable-rdc

    INTERFACE_LINK_DIRS
      lib

    INTERFACE_INSTALL_RPATH_DIRS
      lib
  )
  therock_cmake_subproject_glob_c_sources(rdc
    SUBDIRS .
  )
  therock_cmake_subproject_provide_package(rdc rdc lib/cmake/rdc)
  therock_cmake_subproject_activate(rdc)

  # RDC validation tests (runtime tests excluded - require GPU hardware)

  therock_test_validate_shared_lib(
    PATH rdc/stage/portable-rdc/lib
    LIB_NAMES
      librdc_bootstrap.so
      librdc_client.so
  )

  # Verify gRPC symbols are hidden to prevent ODR violations
  add_test(
    NAME rdc-validate-no-grpc-symbol-pollution
    COMMAND bash -c "! nm -D ${CMAKE_CURRENT_BINARY_DIR}/rdc/stage/portable-rdc/lib/librdc_client.so* 2>/dev/null | grep -q ' T.*grpc::'"
  )
  set_tests_properties(rdc-validate-no-grpc-symbol-pollution PROPERTIES
    LABELS "rdc;symbol-visibility"
  )

  # Verify binaries exist
  add_test(
    NAME rdc-verify-binaries-exist
    COMMAND bash -c "test -x ${CMAKE_CURRENT_BINARY_DIR}/rdc/stage/portable-rdc/bin/rdcd && test -x ${CMAKE_CURRENT_BINARY_DIR}/rdc/stage/portable-rdc/bin/rdci"
  )
  set_tests_properties(rdc-verify-binaries-exist PROPERTIES
    LABELS "rdc;build-verification"
  )

  # Verify library SONAME structure
  add_test(
    NAME rdc-verify-library-sonames
    COMMAND bash -c "test -L ${CMAKE_CURRENT_BINARY_DIR}/rdc/stage/portable-rdc/lib/librdc.so && test -L ${CMAKE_CURRENT_BINARY_DIR}/rdc/stage/portable-rdc/lib/librdc_client.so && test -L ${CMAKE_CURRENT_BINARY_DIR}/rdc/stage/portable-rdc/lib/librdc_bootstrap.so"
  )
  set_tests_properties(rdc-verify-library-sonames PROPERTIES
    LABELS "rdc;build-verification"
  )

  # Provide RDC artifact (portable-rdc/ structure per RFC0007)
  therock_provide_artifact(rdc
    TARGET_NEUTRAL
    DESCRIPTOR artifact-rdc.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
      test
    SUBPROJECT_DEPS
      amdsmi
      rocprofiler-sdk
      rdc
      ${THEROCK_BUNDLED_LIBCAP}
      ${THEROCK_BUNDLED_ZLIB}
  )

endif(THEROCK_ENABLE_RDC)
