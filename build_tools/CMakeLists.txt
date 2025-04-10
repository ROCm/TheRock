add_test(
    NAME build_tools_fileset_tool_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fileset_tool_test.py"
)

add_test(
    NAME build_tools_artifacts_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/artifacts_test.py"
)
