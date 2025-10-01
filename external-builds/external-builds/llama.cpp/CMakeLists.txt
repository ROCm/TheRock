if(THEROCK_ENABLE_LLAMA_CPP)
  ##############################################################################
  # llama.cpp integration
  ##############################################################################

  therock_cmake_subproject_declare(
    llamacpp
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llamacpp"
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DGGML_HIP=ON
      -DAMDGPU_TARGETS=${THEROCK_AMDGPU_TARGETS}
      -DCMAKE_BUILD_TYPE=${THEROCK_BUILD_TYPE}
      -DLLAMA_BUILD_EXAMPLES=ON
      -DHIP_PLATFORM=amd
      # TODO: Potential future implementation of CURL build in TheRock
      -DLLAMA_CURL=OFF
      # TODO: integrate rocWMMA to enable using wmma in llamacpp
      # for RDNA3+ GPUs.
      -DGGML_HIP_ROCWMMA_FATTN=OFF
      -DROCM_PATH="${THEROCK_BINARY_DIR}/dist/rocm"
      -DHIP_PATH="${THEROCK_BINARY_DIR}/dist/rocm"
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      hipBLAS-common
    RUNTIME_DEPS
      hipBLAS
      rocBLAS
  )

  # Activate the llama.cpp subproject for linking
  therock_cmake_subproject_activate(llamacpp)

  # Provide artifacts for TheRock package management
  therock_provide_artifact(
    llamacpp
    DESCRIPTOR artifact-llama-cpp.toml
    COMPONENTS
      dbg
      dev
      lib
      run
    SUBPROJECT_DEPS
      llamacpp
  )
endif()
