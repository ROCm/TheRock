# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT THEROCK_BUILD_TESTING)
  return()
endif()

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
