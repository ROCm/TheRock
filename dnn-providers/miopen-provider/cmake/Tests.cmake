# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(HIPDNN_SKIP_TESTS)
    return()
endif()

include(GoogleTest)

find_package(Python3 COMPONENTS Interpreter)

set(CHECK_DEPENDS_GLOBAL "" CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE)
set(CHECK_EXECUTABLE_PATHS_GLOBAL "" CACHE INTERNAL "Accumulated global check executable paths" FORCE)

# Builds the test environment list with optional code coverage support
# ~~~
# Parameters:
#   OUT_VAR - The name of the variable to store the result in (will be set in PARENT_SCOPE)
# ~~~
function(_build_test_environment_list_internal OUT_VAR)
    set(ENVIRONMENT_LIST "")
    if(DEFINED TEST_ENVIRONMENT)
        set(ENVIRONMENT_LIST ${TEST_ENVIRONMENT})
    endif()

    if(HIPDNN_ENABLE_COVERAGE)
        # Ensure coverage report directory exists
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/coverage_report/profraw")

        # For code coverage builds, we want each profraw file to have a unique name.  The %m in the
        # LLVM_PROFILE_FILE environment variable will auto generate a unique id.
        list(APPEND ENVIRONMENT_LIST
             "LLVM_PROFILE_FILE=${CMAKE_BINARY_DIR}/coverage_report/profraw/%m.profraw"
        )
    endif()

    set(${OUT_VAR} ${ENVIRONMENT_LIST} PARENT_SCOPE)
endfunction() # _build_test_environment_list_internal

# Creates a custom target to validate test names using a Python script
function(_create_test_name_validation_target_internal)
    if(Python3_FOUND)
        # Write list of test executables with their paths to a file
        set(TEST_EXECUTABLES_FILE ${CMAKE_BINARY_DIR}/test_executables.txt)
        list(REMOVE_DUPLICATES CHECK_EXECUTABLE_PATHS_GLOBAL)
        file(WRITE ${TEST_EXECUTABLES_FILE} "")
        foreach(test_executable ${CHECK_EXECUTABLE_PATHS_GLOBAL})
            file(APPEND ${TEST_EXECUTABLES_FILE} "${test_executable}\n")
        endforeach()

        add_custom_command(
            OUTPUT ${CMAKE_BINARY_DIR}/test_names_validated
            COMMAND
                ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py
                --test-executables ${TEST_EXECUTABLES_FILE} --build-dir ${CMAKE_BINARY_DIR} --strict
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/test_names_validated
            DEPENDS ${CMAKE_SOURCE_DIR}/cmake/scripts/test_name_validator.py ${CHECK_DEPENDS_GLOBAL}
            COMMENT "Validating test names with --gtest_list_tests test collection"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            VERBATIM
        )

        add_custom_target(
            validate_test_names DEPENDS ${CMAKE_BINARY_DIR}/test_names_validated
            COMMENT "Validating test names"
        )
    else()
        message(WARNING "Python3 not found. Test name validation will be skipped.")
        add_custom_target(
            validate_test_names COMMAND ${CMAKE_COMMAND} -E echo
                                        "Test name validation skipped - Python3 not found"
            COMMENT "Skipping test name validation"
        )
    endif() # Python3_FOUND
endfunction() # _create_test_name_validation_target_internal

enable_testing() # Cmake wont discover or run tests without this line

# Internal helper function to create a ctest target
# ~~~
# Parameters:
#   TARGET_NAME - Name of the ctest target to create
#   LABEL - Optional label filter for ctest (empty string for no filter)
#   VERBOSE - Set to TRUE to add --verbose flag, FALSE otherwise
#   COMMENT - Comment describing the target
# ~~~
function(_add_ctest_target_internal TARGET_NAME LABEL VERBOSE COMMENT)
    # Build the ctest command
    set(CTEST_CMD ${CMAKE_COMMAND} -E env ${CTEST_ENV} ${CMAKE_CTEST_COMMAND})

    # Add label filter if specified
    if(NOT "${LABEL}" STREQUAL "")
        list(APPEND CTEST_CMD -L "${LABEL}")
    endif()

    # Always add --output-on-failure
    list(APPEND CTEST_CMD --output-on-failure)

    # Add --verbose if requested
    if(VERBOSE)
        list(APPEND CTEST_CMD --verbose)
    endif()

    # Add configuration
    list(APPEND CTEST_CMD -C ${CMAKE_CFG_INTDIR})

    # Create the target
    add_custom_target(${TARGET_NAME} COMMAND ${CTEST_CMD} COMMENT "${COMMENT}" USES_TERMINAL)
    add_dependencies(${TARGET_NAME} validate_test_names)
    message(VERBOSE "Created ${TARGET_NAME} target")
endfunction() # _add_ctest_target_internal

# Internal helper function to create the check targets for running tests via ctest
function(_create_ctest_targets_internal)
    # cmake-format: off
    # Build test environment once for all ctest targets
    _build_test_environment_list_internal(CTEST_ENV)

    # Regular targets (without --verbose)
    _add_ctest_target_internal(check_ctest "" FALSE "Running all tests via ctest")
    _add_ctest_target_internal(unit-check_ctest "unit_test" FALSE "Running unit tests via ctest")
    _add_ctest_target_internal(integration-check_ctest "integration_test" FALSE "Running integration tests via ctest")

    # Verbose targets (with --verbose)
    _add_ctest_target_internal(check_ctest-verbose "" TRUE "Running all tests via ctest (verbose)")
    _add_ctest_target_internal(unit-check_ctest-verbose "unit_test" TRUE "Running unit tests via ctest (verbose)")
    _add_ctest_target_internal(integration-check_ctest-verbose "integration_test" TRUE "Running integration tests via ctest (verbose)")
    # cmake-format: on
endfunction() # create_ctest_targets

# Finalizes and creates all of the test targets
function(finalize_test_targets)
    _create_test_name_validation_target_internal()

    _create_ctest_targets_internal()

    # cmake-format: off
    # Create alias test targets without '_ctest' in the name
    # Regular targets (without --verbose)
    add_custom_target(check DEPENDS check_ctest COMMENT "Running all tests via ctest")
    add_custom_target(unit-check DEPENDS unit-check_ctest COMMENT "Running unit tests via ctest")
    add_custom_target(integration-check DEPENDS integration-check_ctest COMMENT "Running integration tests via ctest")
    message(STATUS "Created ctest targets: check, unit-check, integration-check")
    # Verbose targets (with --verbose)
    add_custom_target(check-verbose DEPENDS check_ctest-verbose COMMENT "Running all tests via ctest (verbose)")
    add_custom_target(unit-check-verbose DEPENDS unit-check_ctest-verbose COMMENT "Running unit tests via ctest (verbose)")
    add_custom_target(integration-check-verbose DEPENDS integration-check_ctest-verbose COMMENT "Running integration tests via ctest (verbose)")
    message(STATUS "Created ctest verbose targets: check-verbose, unit-check-verbose, integration-check-verbose")
    # cmake-format: on
endfunction() # finalize_test_targets

# ~~~
# Internal helper function to record, configure, and register a ctest test target. Assumes that the
# test target is a gtest executable, setting up:
# - Test name validation tracking (adds to global dependency and executable path lists)
# - RPATH settings for relocatable test executables
# - Installation rules for test binaries
# - CTest registration with appropriate labels (e.g. unit / integration test labels)
# Parameters:
#   APPEND_FUNCTION_SUFFIX - Label to apply to the test (e.g., "unit_test", "integration_test", "test")
#   TARGET - Name of the test executable target (must already exist)
#   WORKING_DIR - Working directory for test execution
# ~~~
function(_add_test_target_internal APPEND_FUNCTION_SUFFIX TARGET WORKING_DIR)
    message(STATUS "Appending unclassified check target: ${TARGET} in working directory: ${WORKING_DIR}")

    # Track the dependencies for test name validation
    set(CHECK_DEPENDS_GLOBAL ${CHECK_DEPENDS_GLOBAL} ${TARGET}
        CACHE INTERNAL "Accumulated global dependencies for test name validation" FORCE
    )
    # Track the binary paths for test name validation
    set(CHECK_EXECUTABLE_PATHS_GLOBAL ${CHECK_EXECUTABLE_PATHS_GLOBAL} "${CMAKE_INSTALL_BINDIR}/${TARGET}"
        CACHE INTERNAL "Accumulated global check executable paths" FORCE
    )

    # Track this test target for later use in generating installed CTestTestfile.cmake
    set_property(GLOBAL APPEND PROPERTY HIPDNN_TEST_TARGETS ${TARGET})

    set_target_properties(
        ${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
    )

    # Track this test target for later use in generating installed CTestTestfile.cmake
    set_property(GLOBAL APPEND PROPERTY MIOPEN_LEGACY_TEST_TARGETS ${TARGET})

    # Make test executables relocatable so they can find libraries when build directory is moved
    # Include both the main lib directory and the engine plugin directories
    set_target_properties(
        ${TARGET}
        PROPERTIES
            INSTALL_RPATH
            "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/hipdnn_plugins/engines"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_RPATH_USE_ORIGIN TRUE
    )

    # Install test executables to bin directory
    install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_test(NAME ${TARGET} COMMAND ${TARGET} WORKING_DIRECTORY ${WORKING_DIR})
    set_tests_properties(${TARGET} PROPERTIES LABELS ${APPEND_FUNCTION_SUFFIX})
endfunction() # _add_gtest_target_internal

# Adds a generic test target
function(add_unclassified_test_target TARGET WORKING_DIR)
    _add_test_target_internal(test ${TARGET} ${WORKING_DIR})
endfunction() # add_unclassified_test_target

# Adds a unit test target
function(add_unit_test_target TARGET WORKING_DIR)
    _add_test_target_internal(unit_test ${TARGET} ${WORKING_DIR})
endfunction() # add_unit_test_target

# Adds an integration test target
function(add_integration_test_target TARGET WORKING_DIR)
    _add_test_target_internal(integration_test ${TARGET} ${WORKING_DIR})
endfunction() # add_integration_test_target

# Install CTest configuration files for direct test execution This should be called once at the end
# of the main CMakeLists.txt after all tests are registered
function(install_miopen_legacy_plugin_ctest_files)
    # Define the CTest installation directory
    set(MIOPEN_LEGACY_PLUGIN_CTEST_FILE_INSTALL_PATH "${CMAKE_INSTALL_BINDIR}/miopen_legacy_plugin")

    # Generate a new CTestTestfile.cmake that references installed test executables
    set(INSTALLED_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install")

    file(WRITE "${INSTALLED_CTEST_FILE}"
         "# Autogenerated CTestTestfile for installed miopen_legacy_plugin tests\n"
    )
    file(APPEND "${INSTALLED_CTEST_FILE}" "# Generated by miopen_legacy_plugin build system\n\n")

    # Get all test targets that were registered
    get_property(all_tests GLOBAL PROPERTY MIOPEN_LEGACY_TEST_TARGETS)

    foreach(test_target ${all_tests})
        file(APPEND "${INSTALLED_CTEST_FILE}" "add_test(${test_target} \"../${test_target}\")\n")
    endforeach()

    # Install the generated CTestTestfile.cmake to MIOPEN_LEGACY_PLUGIN_CTEST_FILE_INSTALL_PATH
    install(FILES "${INSTALLED_CTEST_FILE}"
            DESTINATION ${MIOPEN_LEGACY_PLUGIN_CTEST_FILE_INSTALL_PATH} RENAME CTestTestfile.cmake
    )

endfunction() # install_miopen_legacy_plugin_ctest_files
