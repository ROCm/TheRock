# OpenCL Conformance Test Suite (CTS)
# This builds the OpenCL CTS tests for validating OpenCL runtime implementations

if(THEROCK_BUILD_TESTING AND THEROCK_ENABLE_OCL_RUNTIME)
  if(THEROCK_ENABLE_OPENCL_CTS)

    ##############################################################################
    # opencl-cts
    # OpenCL Conformance Test Suite
    ##############################################################################

    therock_cmake_subproject_declare(opencl-cts
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/opencl-cts"
        "${CMAKE_CURRENT_SOURCE_DIR}/opencl-headers"
        "${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/opencl-cts"
      BACKGROUND_BUILD
      CMAKE_ARGS
        "-DCL_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/opencl-headers"
        "-DSPIRV_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers"
        "-DOPENCL_LIBRARIES=OpenCL"
        "-DCMAKE_BUILD_TYPE=Release"
      BUILD_DEPS
        rocm-cmake
      RUNTIME_DEPS
        ocl-clr
        ocl-icd
        ROCR-Runtime
      INTERFACE_LINK_DIRS
        "lib"
      INTERFACE_INSTALL_RPATH_DIRS
        "lib"
    )
    therock_cmake_subproject_glob_c_sources(opencl-cts SUBDIRS .)
    therock_cmake_subproject_activate(opencl-cts)

    therock_provide_artifact(opencl-cts
      TARGET_NEUTRAL
      DESCRIPTOR artifact-opencl-cts.toml
      COMPONENTS
        test
      SUBPROJECT_DEPS
        opencl-cts
    )

  endif(THEROCK_ENABLE_OPENCL_CTS)
endif(THEROCK_BUILD_TESTING AND THEROCK_ENABLE_OCL_RUNTIME)
