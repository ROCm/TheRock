therock_subproject_fetch(therock-dyninst-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/dyninst/dyninst/archive/refs/tags/v13.0.0.tar.gz
  URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/dyninst-13.0.0.tar.gz
  URL_HASH SHA256=1bc48d26478b677a6c090c25586a447507bd1b4cf88d369bd61820005ce1be39
)

therock_cmake_subproject_declare(therock-dyninst
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  RUNTIME_DEPS
    therock-boost
    therock-tbb
    ${THEROCK_BUNDLED_BINUTILS}
    ${THEROCK_BUNDLED_ELFUTILS}
  INSTALL_RPATH_DIRS
    lib/rocm_sysdeps/lib
)
therock_cmake_subproject_provide_package(therock-dyninst Dyninst lib/cmake)
therock_cmake_subproject_activate(therock-dyninst)

add_dependencies(therock-third-party therock-dyninst)

therock_provide_artifact(dyninst
  DESCRIPTOR artifact_dyninst.toml
  TARGET_NEUTRAL
  COMPONENTS
    dbg
    dev
    lib
    run
  SUBPROJECT_DEPS
    therock-dyninst
)
