################################################################################
# mxDataGenerator
################################################################################

if(THEROCK_USE_ROCM_LIBRARIES)
  set(_mxdatagenerator_source_dir "${THEROCK_ROCM_LIBRARIES_SOURCE_DIR}/shared/mxdatagenerator")
else()
  set(_mxdatagenerator_source_dir "mxDataGenerator")
endif()

therock_cmake_subproject_declare(mxDataGenerator
  EXTERNAL_SOURCE_DIR "${_mxdatagenerator_source_dir}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/mxDataGenerator"
  BACKGROUND_BUILD
  CMAKE_ARGS
    # TODO: Enable tests once there is install support
    #-DMXDATAGENERATOR_BUILD_TESTING=${THEROCK_BUILD_TESTING}
  COMPILER_TOOLCHAIN
    amd-hip
  BUILD_DEPS
    therock-googletest
)
therock_cmake_subproject_provide_package(mxDataGenerator
  mxDataGenerator lib/cmake)
therock_cmake_subproject_activate(mxDataGenerator)

################################################################################
# support artifact
################################################################################

therock_provide_artifact(support
  DESCRIPTOR artifact-support.toml
  COMPONENTS
    dev
    doc
  SUBPROJECT_DEPS
    mxDataGenerator
)
