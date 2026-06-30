# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Workaround: rocPRIM benchmarks link against amd_smi but the project never
# calls find_package(amd_smi).  Without the imported target the include
# directories are not propagated and the build fails with a missing
# <amd_smi/amdsmi.h> header.  Injecting the find_package call here via
# CMAKE_INCLUDES ensures the target exists before the benchmark subdirectory
# is processed.
#
# TODO(https://github.com/ROCm/TheRock/issues/4651): Remove once upstream
# rocPRIM adds its own find_package(amd_smi).

if(NOT TARGET amd_smi)
  find_package(amd_smi REQUIRED CONFIG)
endif()
