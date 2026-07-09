# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# These tests install to libexec/rocprofiler-compute/tests/ instead of the
# default bin/. Tell TheRock the actual origin so it can compute correct
# relative RPATH entries (including sysdeps under lib/rocm_sysdeps/lib).
foreach(_test_target test-rocprofiler-compute-tool test-pc-sampling-collector)
  if(TARGET ${_test_target})
    set_target_properties(${_test_target} PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN libexec/rocprofiler-compute/tests
    )
  endif()
endforeach()
