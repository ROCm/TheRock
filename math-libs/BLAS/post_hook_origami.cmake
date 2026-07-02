# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# origami (in rocm-libraries) generates its own CTestTestfile.cmake that runs
# tests via bare `python` and with build-tree-relative paths. Neither survives
# packaging: `python` may be absent in the test environment, and the paths point
# at origami's build directory rather than the unpacked artifact. This cannot be
# fixed in origami itself, because the correct test paths depend on TheRock's
# install layout, which is only known once the artifact is unpacked. So we
# install a replacement CTestTestfile here that uses `python3` and paths
# relative to ${CMAKE_CURRENT_LIST_DIR}.
if(NOT THEROCK_BUILD_TESTING)
  return()
endif()

# Written as a bracket string so ${CMAKE_CURRENT_LIST_DIR} is emitted literally
# and expanded by ctest when it reads the installed file, not expanded here.
set(_ctest_content [=[
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_test(origami-tests "${CMAKE_CURRENT_LIST_DIR}/../origami-tests")
set_tests_properties(origami-tests PROPERTIES
    LABELS "cpp"
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)

add_test(origami_python_tests python3 -m pytest "${CMAKE_CURRENT_LIST_DIR}/tests" -v)
set_tests_properties(origami_python_tests PROPERTIES
    LABELS "python"
    SKIP_RETURN_CODE 5
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
)
]=])

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.therock" "${_ctest_content}")

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.therock"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/origami"
    COMPONENT tests
    RENAME CTestTestfile.cmake
)
