# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# These tests install to libexec/rocprofiler-compute/tests/ instead of the
# default bin/. Tell TheRock the actual origin so it can compute correct
# relative RPATH entries (including sysdeps under lib/rocm_sysdeps/lib).
foreach(_test_target
        test-rocprofiler-compute-tool
        test-pc-sampling-collector
        test-compression
        test-roctx-recordfn)
  if(TARGET ${_test_target})
    set_target_properties(${_test_target} PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN libexec/rocprofiler-compute/tests
    )
  endif()
endforeach()

# librocprofiler-compute-tool.so installs to lib/rocprofiler-compute/, not the
# default lib/, so its sysdeps entry otherwise lands a directory short.
if(TARGET rocprofiler-compute-tool)
  set_target_properties(rocprofiler-compute-tool PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN lib/rocprofiler-compute
  )
endif()
