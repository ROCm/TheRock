# OpenCL Conformance Test Suite (CTS)
# This builds the OpenCL CTS tests for validating OpenCL runtime implementations

if(THEROCK_BUILD_TESTING AND THEROCK_ENABLE_OCL_RUNTIME)
  if(THEROCK_ENABLE_OPENCL_CTS)

    ##############################################################################
    # opencl-icd-loader
    # OpenCL ICD Loader with OpenCL 3.0 support for CTS
    ##############################################################################

    therock_cmake_subproject_declare(opencl-icd-loader
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/opencl-icd-loader"
        "${CMAKE_CURRENT_SOURCE_DIR}/opencl-headers"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/opencl-icd-loader"
      BACKGROUND_BUILD
      CMAKE_ARGS
        "-DOPENCL_ICD_LOADER_HEADERS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/opencl-headers"
        "-DOPENCL_ICD_LOADER_BUILD_SHARED_LIBS=ON"
        "-DCMAKE_BUILD_TYPE=Release"
      BUILD_DEPS
        rocm-cmake
      INTERFACE_LINK_DIRS
        "lib"
      INTERFACE_INSTALL_RPATH_DIRS
        "lib"
    )
    therock_cmake_subproject_glob_c_sources(opencl-icd-loader SUBDIRS .)
    therock_cmake_subproject_activate(opencl-icd-loader)

    ##############################################################################
    # spirv-tools
    # SPIR-V Tools for OpenCL CTS (spirv-as, spirv-val)
    ##############################################################################

    therock_cmake_subproject_declare(spirv-tools
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/spirv-tools"
        "${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/spirv-tools"
      BACKGROUND_BUILD
      CMAKE_ARGS
        "-DSPIRV-Headers_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers"
        "-DSPIRV_SKIP_TESTS=ON"
        "-DSPIRV_SKIP_EXECUTABLES=OFF"
        "-DCMAKE_BUILD_TYPE=Release"
      BUILD_DEPS
        rocm-cmake
      INTERFACE_PROGRAM_DIRS
        "bin"
    )
    therock_cmake_subproject_glob_c_sources(spirv-tools SUBDIRS .)
    therock_cmake_subproject_activate(spirv-tools)

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
        "-DCL_LIB_DIR=${CMAKE_CURRENT_BINARY_DIR}/opencl-icd-loader/dist/lib"
        "-DSPIRV_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/spirv-headers"
        "-DSPIRV_TOOLS_DIR=${CMAKE_CURRENT_BINARY_DIR}/spirv-tools/dist/bin"
        "-DOPENCL_LIBRARIES=OpenCL"
        "-DCMAKE_BUILD_TYPE=Release"
      BUILD_DEPS
        rocm-cmake
        spirv-tools
        opencl-icd-loader
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
