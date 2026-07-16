# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Init hook for 32-bit OCL runtime build
# This is loaded via CMAKE_PROJECT_TOP_LEVEL_INCLUDES

# The CLR project calls find_package(amd_comgr) but we're providing amd-comgr32
# Add "amd_comgr" to the provided packages list so the dependency provider allows it
if(DEFINED THEROCK_PROVIDED_PACKAGES)
  list(APPEND THEROCK_PROVIDED_PACKAGES "amd_comgr")
endif()

