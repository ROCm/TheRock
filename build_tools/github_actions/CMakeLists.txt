add_test(
    NAME build_tools_github_actions_configure_ci_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/configure_ci_test.py"
)

add_test(
    NAME build_tools_github_actions_configure_target_run_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/configure_target_run_test.py"
)

add_test(
    NAME build_tools_github_actions_determine_version_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/determine_version_test.py"
)

add_test(
    NAME build_tools_github_actions_fetch_package_targets_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/fetch_package_targets_test.py"
)

add_test(
    NAME build_tools_github_actions_python_to_cp_version_test
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_to_cp_version_test.py"
)
