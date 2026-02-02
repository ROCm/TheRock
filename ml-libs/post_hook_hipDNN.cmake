# Add the test plugin directories to the private install RPATH dirs
list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib/test_plugins/custom")
list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib/test_plugins/default")

# Set origin properly for the RPATH to work
set_target_properties(
    test_good_plugin
    test_execute_fails_plugin
    test_no_applicable_engines_a_plugin
    test_no_applicable_engines_b_plugin
    test_duplicate_id_a_plugin
    test_duplicate_id_b_plugin
    test_incomplete_api_plugin
    test_knobs_plugin
    test_knob_constraint_validation_plugin
    PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/test_plugins/custom"
)

set_target_properties(
    test_good_default_plugin
    PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/test_plugins/default"
)
