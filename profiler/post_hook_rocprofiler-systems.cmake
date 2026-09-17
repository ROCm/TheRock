# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Dyninst and rocprofiler-systems internal libraries live in lib/rocprofiler-systems.
# Set the RPATH origin per-target so that (for example) lib/rocm_sysdeps/lib resolves as
# $ORIGIN/../rocm_sysdeps/lib rather than $ORIGIN/rocm_sysdeps/lib.
set(_rocprofsys_lib_targets
  common
  dynElf
  dynDwarf
  dyninstAPI
  dyninstAPI_RT
  dynC_API
  instructionAPI
  parseAPI
  patchAPI
  pcontrol
  stackwalk
  symtabAPI
  gotcha
)

foreach(_target ${_rocprofsys_lib_targets})
  if(TARGET "${_target}")
    set_target_properties("${_target}" PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN lib/rocprofiler-systems
    )
  endif()
endforeach()

# rocprof-sys-unit-tests is a native, mock-backed unit binary. Upstream builds
# it but does not install it, so the ordinary test artifact currently contains
# only the pytest/CTest integration payload. Package the binary explicitly for
# the CPU-only sanitizer lane and give TheRock its real install origin so the
# ROCm, Dyninst, sysdeps, and compiler-rt RPATH entries are relocatable.
if(TARGET rocprof-sys-unit-tests)
  set_target_properties(rocprof-sys-unit-tests PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN
      share/rocprofiler-systems/tests/unit-tests/bin
  )
  install(TARGETS rocprof-sys-unit-tests
    RUNTIME DESTINATION share/rocprofiler-systems/tests/unit-tests/bin
    COMPONENT rocprofiler-systems-tests
  )
endif()

# The upstream unit-test aggregate links the shared GoogleTest build. Those
# libraries are private test dependencies and are not installed by upstream,
# so installing only rocprof-sys-unit-tests leaves a binary that the dynamic
# loader cannot start. Keep the private runtime with the test artifact in the
# already-searched rocprofiler-systems library directory.
foreach(_target gtest gtest_main gmock)
  if(TARGET "${_target}")
    set_target_properties("${_target}" PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN lib/rocprofiler-systems
    )
    install(TARGETS "${_target}"
      LIBRARY DESTINATION lib/rocprofiler-systems
      COMPONENT rocprofiler-systems-tests
    )
  endif()
endforeach()
