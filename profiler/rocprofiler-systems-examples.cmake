# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT THEROCK_ENABLE_ROCPROFSYS OR NOT THEROCK_BUILD_TESTING)
  return()
endif()

set(_rocprofiler_systems_examples_runtime_deps
  rocprofiler-systems
)
set(_ROCPROFSYS_DISABLED_EXAMPLES
  lulesh
  openmp-target
  causal
)
if(THEROCK_ENABLE_ROCSHMEM)
  list(APPEND _rocprofiler_systems_examples_runtime_deps rocshmem)
else()
  list(APPEND _ROCPROFSYS_DISABLED_EXAMPLES rocshmem)
endif()

# Escape semicolons in lists to prevent shell interpretation.
if(THEROCK_TEST_AMDGPU_TARGETS STREQUAL "THEROCK_TEST_AMDGPU_TARGETS-NOTFOUND")
  set(_ROCPROFSYS_GFX_TARGETS_ESCAPED "")
else()
  string(REPLACE ";" "\;" _ROCPROFSYS_GFX_TARGETS_ESCAPED
    "${THEROCK_TEST_AMDGPU_TARGETS}"
  )
endif()
string(REPLACE ";" "\;" _ROCPROFSYS_DISABLED_EXAMPLES_ESCAPED
  "${_ROCPROFSYS_DISABLED_EXAMPLES}"
)

# TODO: This should be built as target-specific, but that is not currently possible because
# the artifact splitter strips per-arch HIP offload bundles out of the example binaries,
# preventing llvm-objdump (used in our tests) from detecting the offload architectures.
# See https://github.com/ROCm/TheRock/issues/4848.
therock_cmake_subproject_declare(rocprofiler-systems-examples
  USE_TEST_AMDGPU_TARGETS
  EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprofiler-systems/examples"
  BINARY_DIR "${CMAKE_BINARY_DIR}/profiler/rocprofiler-systems-examples"
  BACKGROUND_BUILD
  CMAKE_ARGS
    -DROCPROFSYS_GFX_TARGETS="${_ROCPROFSYS_GFX_TARGETS_ESCAPED}"
    -DROCPROFSYS_DISABLE_EXAMPLES="${_ROCPROFSYS_DISABLED_EXAMPLES_ESCAPED}"
  COMPILER_TOOLCHAIN
    amd-hip
  INSTALL_RPATH_DIRS
    "lib"
    "lib/rocprofiler-systems"
    "lib/llvm/lib"
  RUNTIME_DEPS
    ${_rocprofiler_systems_examples_runtime_deps}
)
therock_cmake_subproject_glob_c_sources(rocprofiler-systems-examples
  SUBDIRS .
)
therock_cmake_subproject_activate(rocprofiler-systems-examples)

therock_provide_artifact(rocprofiler-systems-examples
  TARGET_NEUTRAL
  DESCRIPTOR "${CMAKE_SOURCE_DIR}/profiler/artifact-rocprofiler-systems-examples.toml"
  COMPONENTS
    test
  SUBPROJECT_DEPS
    rocprofiler-systems-examples
)
