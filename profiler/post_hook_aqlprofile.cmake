# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# AQLProfile builds native unit tests when AQLPROFILE_BUILD_TESTS is enabled,
# but upstream's tests install component contains only the GPU integration-test
# sources. Install the exact native binaries used by the device-free sanitizer
# lane and keep their runtime search paths relocatable in an artifact prefix.
if(AQLPROFILE_BUILD_TESTS)
  set(_aqlprofile_host_test_manifest
    "${CMAKE_CURRENT_LIST_DIR}/aqlprofile_host_tsan_tests.json"
  )
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_aqlprofile_host_test_manifest}"
  )
  set(_aqlprofile_host_test_targets
    gfx9-memory-manager-test
    aqlprofile-test
    command-buffer-test
    counters-test
    pm4-factory-test
    logger-test
    aql-profile-v2-test
    aql-profile-v2-c-compatibility-test
    command-builder-test
    pmc-builder-test
    gfx9-command-builder-test
    spm-builder-test
    trace-config-test
    sqtt-builder-test
    utility_tests
  )

  foreach(_target IN LISTS _aqlprofile_host_test_targets)
    if(NOT TARGET "${_target}")
      message(FATAL_ERROR
        "Required AQLProfile host-test target is missing: ${_target}"
      )
    endif()
    set_target_properties("${_target}" PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN
        share/hsa-amd-aqlprofile/tests/host-tsan/bin
    )
  endforeach()

  install(TARGETS ${_aqlprofile_host_test_targets}
    RUNTIME DESTINATION share/hsa-amd-aqlprofile/tests/host-tsan/bin
    COMPONENT tests
  )
  install(FILES "${_aqlprofile_host_test_manifest}"
    DESTINATION share/hsa-amd-aqlprofile/tests/host-tsan
    RENAME host_tsan_tests.json
    COMPONENT tests
  )
endif()
