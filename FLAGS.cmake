# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# FLAGS.cmake
# Central registry of build flags for TheRock.
#
# Each flag creates a THEROCK_FLAG_${NAME} cache variable that can be
# controlled via -DTHEROCK_FLAG_<NAME>=ON|OFF on the cmake command line.
#
# See docs/development/flags.md for documentation on this system.

include(therock_flag_utils)

###############################################################################
# Flag declarations
###############################################################################

# Build flag infrastructure conformance canaries. These are consumed by the
# unconditional aux-overlay compile check. They deliberately exercise both
# BOOL values and INTEGER serialization. Do not remove or repurpose them.
therock_declare_flag(
  NAME ROCM_BUILD_FLAGS_CANARY_BOOL_FALSE
  TYPE BOOL
  DEFAULT_VALUE OFF
  DESCRIPTION "Build flag infrastructure false-value conformance canary"
)

therock_declare_flag(
  NAME ROCM_BUILD_FLAGS_CANARY_BOOL_TRUE
  TYPE BOOL
  DEFAULT_VALUE ON
  DESCRIPTION "Build flag infrastructure true-value conformance canary"
)

therock_declare_flag(
  NAME ROCM_BUILD_FLAGS_CANARY_INTEGER_NEGATIVE
  TYPE INTEGER
  DEFAULT_VALUE -17
  VALID_VALUES -17
  DESCRIPTION "Build flag infrastructure integer conformance canary"
)

therock_declare_flag(
  NAME KPACK_SPLIT_ARTIFACTS
  DEFAULT_VALUE ON
  DESCRIPTION "Split target-specific artifacts into generic and arch-specific components"
)

therock_declare_flag(
  NAME HIPDNN_ENABLE_SDPA
  DEFAULT_VALUE OFF
  DESCRIPTION "Enable SDPA (Scaled Dot-Product Attention) support in hipDNN"
  CMAKE_VARS
    HIPDNN_ENABLE_SDPA=ON
  SUB_PROJECTS
    hipDNN
)

therock_declare_flag(
  NAME HIPBLASLTPROVIDER_ENABLE_MX_GEMM
  DEFAULT_VALUE OFF
  DESCRIPTION "Enable MX (microscaling) data-type support for GEMM in the hipDNN hipBLASLt provider"
  CMAKE_VARS
    HIPBLASLTPROVIDER_ENABLE_MX_GEMM=ON
  SUB_PROJECTS
    hipblasltprovider
)

therock_declare_flag(
  NAME HIPKERNELPROVIDER_ENABLE_ROCKE
  DEFAULT_VALUE OFF
  DESCRIPTION "Build the rocKE engine and smoke tests in hip-kernel-provider"
  CMAKE_VARS
    HIPKERNELPROVIDER_ENABLE_ROCKE=ON
  SUB_PROJECTS
    hipkernelprovider
)

# Gates the generic kernel ingestor platform described in RFC 0017:
# rocm-libraries/projects/hipdnn/docs/rfcs/0017_UniversalKernelDescriptor.md
therock_declare_flag(
  NAME HIPDNN_ENABLE_KERNEL_INGESTOR
  DEFAULT_VALUE OFF
  DESCRIPTION "Enable the generic kernel ingestor build-time logic in hipDNN (dynamic engine loading, kpack bundling/packaging) and its providers"
  CMAKE_VARS
    HIPDNN_ENABLE_KERNEL_INGESTOR=ON
  SUB_PROJECTS
    hipDNN
    hipkernelprovider
)

therock_declare_flag(
  NAME MIOPEN_ENABLE_HIPDNN_WRAPPER
  DEFAULT_VALUE OFF
  DESCRIPTION "Build MIOpen as a public wrapper (libMIOpen.so) over a private implementation library (libMIOpen_private.so), with optional runtime forwarding to hipDNN. See rocm-libraries MIOpen RFC 0001."
  CMAKE_VARS
    MIOPEN_ENABLE_HIPDNN_WRAPPER=ON
  SUB_PROJECTS
    MIOpen
)

therock_declare_flag(
  NAME STAMP_LIBRARY_GIT_VERSIONS
  DEFAULT_VALUE OFF
  DESCRIPTION "Stamp library git revisions into generated version metadata"
  ISSUE https://github.com/ROCm/TheRock/issues/5009
  GLOBAL_PROPAGATE_FLAG
)

therock_declare_flag(
  NAME INCLUDE_HRX
  DEFAULT_VALUE OFF
  DESCRIPTION "Include experimental HRX runtime in core-runtime"
)

therock_declare_flag(
  NAME HSA_WINDOWS_SHARED_RUNTIME
  DEFAULT_VALUE OFF
  DESCRIPTION "Emit ROCR-Runtime and rocminfo from core-runtime on Windows"
)

therock_declare_flag(
  NAME LLVM_ENABLE_ASSERTIONS
  DEFAULT_VALUE OFF
  DESCRIPTION "Build amd-llvm with LLVM assertions enabled. In CI costs roughly 20% more build time overall and up to 35% on device-heavy stages."
  ISSUE "https://github.com/ROCm/TheRock/pull/6102"
  CMAKE_VARS
    LLVM_ENABLE_ASSERTIONS=ON
  SUB_PROJECTS
    amd-llvm
)

###############################################################################
# Branch-specific flag overrides.
# BRANCH_FLAGS.cmake is .gitignored on main but can be committed on
# integration branches to change default flag values via
# therock_override_flag_default().
###############################################################################
include("${CMAKE_CURRENT_SOURCE_DIR}/BRANCH_FLAGS.cmake" OPTIONAL)
include("${CMAKE_CURRENT_BINARY_DIR}/cmake/therock_branch_config.cmake" OPTIONAL)
if(COMMAND therock_apply_branch_config_flags)
  therock_apply_branch_config_flags()
endif()

###############################################################################
# Finalize all flags and report.
###############################################################################
therock_finalize_flags()
therock_report_flags()
