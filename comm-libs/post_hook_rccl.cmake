# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RCCL's host-only microtests are deliberately separate from its GPU-backed
# rccl-UnitTests suite. Install the exact targets admitted by the host-TSAN
# lane so the test artifact can execute them without a source/build tree.
if(BUILD_TESTS)
  set(_therock_rccl_host_test_targets
    rccl-UnitTestsMicro
    rccl-UnitTestsMicroEnqueue
    rccl-UnitTestsMicroInit
    rccl-UnitTestsMicroInit-faultinj
    rccl-UnitTestsMicroInit-uncached
    rccl-UnitTestsNetTelemetry
  )
  foreach(_target IN LISTS _therock_rccl_host_test_targets)
    if(NOT TARGET "${_target}")
      message(FATAL_ERROR
        "Required RCCL host-only test target is missing: ${_target}")
    endif()
  endforeach()
  install(TARGETS ${_therock_rccl_host_test_targets}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT tests
  )
endif()
