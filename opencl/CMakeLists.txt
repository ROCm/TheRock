# OpenCL Conformance Test Suite (CTS)
# This builds the OpenCL CTS tests for validating OpenCL runtime implementations

if(THEROCK_BUILD_TESTING AND THEROCK_ENABLE_OCL_RUNTIME)
  if(THEROCK_ENABLE_OPENCL_CTS)

    ##############################################################################
    # opencl-cts
    # OpenCL Conformance Test Suite - uses distro-provided OpenCL and SPIR-V deps
    ##############################################################################

    find_package(OpenCL REQUIRED)
    find_program(SPIRV_AS_PROGRAM spirv-as REQUIRED)
    get_filename_component(_spirv_tools_dir "${SPIRV_AS_PROGRAM}" DIRECTORY)
    find_path(SPIRV_INCLUDE_DIR NAMES spirv/unified1/spirv.h REQUIRED)

    get_filename_component(_cl_lib_dir "${OpenCL_LIBRARY}" DIRECTORY)

    therock_cmake_subproject_declare(opencl-cts
      USE_DIST_AMDGPU_TARGETS
      EXTERNAL_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/opencl-cts"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/opencl-cts"
      BACKGROUND_BUILD
      CMAKE_ARGS
        "-DCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIRS}"
        "-DCL_LIB_DIR=${_cl_lib_dir}"
        "-DSPIRV_INCLUDE_DIR=${SPIRV_INCLUDE_DIR}"
        "-DSPIRV_TOOLS_DIR=${_spirv_tools_dir}"
        "-DOPENCL_LIBRARIES=OpenCL"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKE_INSTALL_BINDIR=share/opencl/opencl-cts"
      BUILD_DEPS
        rocm-cmake
      RUNTIME_DEPS
        ocl-clr
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
