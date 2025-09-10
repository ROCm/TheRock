add_subdirectory(github_actions)

add_test(
    NAME build_tools_artifacts_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/artifacts_test.py"
)

add_test(
    NAME build_tools_fetch_artifacts_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fetch_artifacts_test.py"
)

add_test(
    NAME build_tools_fileset_tool_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fileset_tool_test.py"
)

add_test(
    NAME build_tools_setup_venv_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/setup_venv_test.py"
)
