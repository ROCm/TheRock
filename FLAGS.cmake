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

therock_declare_flag(
  NAME KPACK_SPLIT_ARTIFACTS
  DEFAULT_VALUE ON
  DESCRIPTION "Split target-specific artifacts into generic and arch-specific components"
)

therock_declare_flag(
  NAME HIP_KERNEL_PROVIDER_ENABLE
  DEFAULT_VALUE OFF
  DESCRIPTION "Enable hip-kernel-provider plugin"
  CMAKE_VARS
    HIP_KERNEL_PROVIDER_ENABLE=ON
  SUB_PROJECTS
    hipkernelprovider
)

therock_declare_flag(
  NAME INCLUDE_HRX
  DEFAULT_VALUE OFF
  DESCRIPTION "Include experimental HRX runtime in core-runtime"
)

# Default OFF: when libLLVM.so exists in the install, processes that also
# load another LLVM (e.g. rocMLIR statically linked into MIGraphX) crash at
# _dl_init from cl::opt singleton conflicts. The Comgr-side COMGR_STATIC_LLVM
# option doesn't fully prevent this -- libamd_comgr.so still ends up with
# DT_NEEDED libLLVM.so via clang/lld INTERFACE_LINK_LIBRARIES + symbol
# versioning. Forced OFF on Windows. See LCOMPILER-2156.
therock_declare_flag(
  NAME LLVM_DYLIB
  DEFAULT_VALUE OFF
  DESCRIPTION "Build LLVM as a shared dylib (libLLVM.so) and link tools against it"
  ISSUE "https://amd-hub.atlassian.net/browse/LCOMPILER-2156"
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
