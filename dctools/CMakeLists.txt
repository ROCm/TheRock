# Data Center Tools

if(THEROCK_ENABLE_RDC)
  # RDC builds with statically linked gRPC; tools (protoc, grpc_cpp_plugin) come from the bundled
  # therock-grpc stage dir to keep symbol visibility and dependencies consistent with RFC0007.
  if(NOT TARGET ${THEROCK_BUNDLED_GRPC})
    message(FATAL_ERROR "RDC requires bundled gRPC (${THEROCK_BUNDLED_GRPC}); enable THEROCK_BUNDLE_SYSDEPS=ON to build gRPC and its dependencies.")
  endif()
  get_target_property(_therock_grpc_stage ${THEROCK_BUNDLED_GRPC} THEROCK_STAGE_DIR)
  if(NOT _therock_grpc_stage)
    message(FATAL_ERROR "Bundled gRPC target ${THEROCK_BUNDLED_GRPC} is missing THEROCK_STAGE_DIR")
  endif()
  set(_therock_grpc_root "${_therock_grpc_stage}/lib/rocm_sysdeps")
  set(_rdc_grpc_cmake_dir "${_therock_grpc_root}/lib/cmake/grpc")
  set(_rdc_protobuf_cmake_dir "${_therock_grpc_root}/lib/cmake/protobuf")

  set(_rdc_runtime_deps
    ROCR-Runtime
    amdsmi
    rocprofiler-sdk
    ${THEROCK_BUNDLED_LIBCAP}
    ${THEROCK_BUNDLED_ZLIB}
    ${THEROCK_BUNDLED_GRPC}
  )

  set(_rdc_artifact_deps
    amdsmi
    rocprofiler-sdk
    rdc
    ${THEROCK_BUNDLED_LIBCAP}
    ${THEROCK_BUNDLED_ZLIB}
    ${THEROCK_BUNDLED_GRPC}
  )

  therock_cmake_subproject_declare(rdc
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rdc"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/rdc"
    BACKGROUND_BUILD

    CMAKE_ARGS
      -DBUILD_PROFILER=ON
      -DBUILD_STANDALONE=ON
      -DBUILD_RUNTIME=ON
      -DBUILD_RVS=OFF
      -DBUILD_TESTS=${THEROCK_BUILD_TESTING}
      -DHIP_PLATFORM=amd
      -DCMAKE_CXX_STANDARD=17
      -DGRPC_ROOT=${_therock_grpc_root}
      -DCMAKE_PREFIX_PATH=${_therock_grpc_root}
      -DgRPC_DIR=${_rdc_grpc_cmake_dir}
      -Dprotobuf_DIR=${_rdc_protobuf_cmake_dir}

    BUILD_DEPS
      amd-llvm

    RUNTIME_DEPS
      ${_rdc_runtime_deps}

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
      ${_rdc_artifact_deps}
  )

endif(THEROCK_ENABLE_RDC)
