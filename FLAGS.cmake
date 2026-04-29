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

###############################################################################
# Branch-specific flag overrides.
# BRANCH_FLAGS.cmake is .gitignored on main but can be committed on
# integration branches to change default flag values via
# therock_override_flag_default().
###############################################################################
include("${CMAKE_CURRENT_SOURCE_DIR}/BRANCH_FLAGS.cmake" OPTIONAL)

###############################################################################
# Finalize all flags and report.
###############################################################################
therock_finalize_flags()
therock_report_flags()
