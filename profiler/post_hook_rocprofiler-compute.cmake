# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# test-rocprofiler-compute-tool installs to libexec/rocprofiler-compute/tests/
# instead of the default bin/. Tell TheRock the actual origin so it can compute
# correct relative RPATH entries.
if(TARGET test-rocprofiler-compute-tool)
  find_package(Threads REQUIRED)
  target_link_libraries(test-rocprofiler-compute-tool PRIVATE Threads::Threads)
  set_target_properties(test-rocprofiler-compute-tool PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN libexec/rocprofiler-compute/tests
  )
endif()
