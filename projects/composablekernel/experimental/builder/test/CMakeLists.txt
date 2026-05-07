# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# CK Builder Test Suite
################################################################################
#
# This file defines the test suite for the Composable Kernel (CK) Builder,
# which is responsible for generating optimized GPU kernels for convolution
# operations.
#
# TESTING PHILOSOPHY:
# -------------------
# Tests are organized into two main categories:
#
# 1. SMOKE TESTS (fast, < 1 second total)
#    - Unit tests that verify the builder's internal logic
#    - Do NOT compile GPU kernels (fast compilation)
#    - Run these frequently during development for quick feedback
#    - Target: `ninja smoke-builder`
#
# 2. REGRESSION TESTS (slower, may take minutes)
#    - Integration tests that compile and verify actual GPU kernels
#    - Ensure the builder generates valid, compilable code
#    - Include expensive "factory tests" that build all MIOpen kernels
#    - Run these before submitting changes
#    - Target: `ninja regression-builder`
#
# QUICK START:
# ------------
# - During development:     ninja smoke-builder
# - Before submitting:      ninja regression-builder
# - Run everything:         ninja check-builder
# - Build specific test:    ninja test_ckb_conv_builder && bin/test_ckb_conv_builder
#
################################################################################

include(gtest)

################################################################################
# Helper Functions
################################################################################

# Helper function to create a gtest executable with common properties
# All builder tests share the same compilation settings and dependencies
function(add_ck_builder_test test_name)
    add_executable(${test_name} ${ARGN} testing_utils.cpp)
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    target_include_directories(${test_name} PRIVATE
        "${PROJECT_SOURCE_DIR}/experimental/builder/include"
        "${PROJECT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_compile_options(${test_name} PRIVATE
        -Wno-global-constructors
        -Wno-c++20-compat
    )
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main GTest::gmock)
endfunction()

# Factory tests attempt to build all the kernels needed by MIOpen.
# These are only for regression testing and development; the builds are too
# expensive for regular use in CI.
function(add_ck_factory_test test_name)
    add_ck_builder_test(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE composablekernels::device_conv_operations)
endfunction()

################################################################################
# SMOKE TESTS - Fast Unit Tests (No Kernel Compilation)
################################################################################
# These tests verify the builder's internal logic without compiling GPU kernels.
# They should complete in under 10ms each and are suitable for frequent execution
# during development.
add_ck_builder_test(test_ckb_conv_builder
    test_bwd_weight_instance_traits.cpp
    test_conv_builder.cpp
    test_fwd_instance_traits.cpp
    test_bwd_data_instance_traits.cpp
    test_instance_traits_util.cpp
    unit_device_buffer.cpp
    unit_tensor_descriptor.cpp
    unit_tensor_foreach.cpp
    unit_error.cpp
    unit_validation.cpp
    unit_debug.cpp
    unit_conv_elementwise_op.cpp
    unit_conv_tensor_layout.cpp
    unit_conv_tensor_type.cpp
    unit_conv_thread_block.cpp
    unit_conv_tuning_params.cpp)

# Tests the inline diff utility used for comparing strings in tests assertions
add_ck_builder_test(test_ckb_inline_diff test_inline_diff.cpp)

# GPU reference validation tests (in validation/ folder)
# 1. Reference kernel execution and InstanceTraits
add_ck_builder_test(test_ckb_reference_execution
    validation/test_reference_execution.cpp
    validation/test_reference_instance_traits.cpp)
target_link_libraries(test_ckb_reference_execution PRIVATE utility)

# Note: Optimized kernel validation tests will be added after merging dev branch
# with kernel Run() implementation from colleague's work

# Tests convolution trait selection and configuration
add_ck_builder_test(test_ckb_conv_traits
    conv/ck/test_conv_traits.cpp
    conv/ck/unit_instance_to_conv_traits.cpp)

# Tests convolution problem description and parameter handling
add_ck_builder_test(test_ckb_conv_description
    test_conv_description.cpp)

################################################################################
# REGRESSION TESTS - Integration Tests (With Kernel Compilation)
################################################################################
# These tests compile actual GPU kernels to verify the builder generates valid,
# compilable code. They are more expensive but catch real-world issues.


# Verifies that GetInstanceString() methods and other functions produce valid kernel code.
# Tests various convolution types:
#   - Group convolution (v3, standard, large tensor, WMMA, DL variants)
#   - Backward weight group convolution (XDL)
# Requires kernel compilation to validate the generated strings through the base class.

set(INSTANCE_STRING_TESTS
    test_instance_string_fwd_grp_conv_v3.cpp
    test_instance_string_fwd_grp_conv.cpp
    test_instance_string_fwd_grp_conv_large_tensor.cpp
    test_instance_string_fwd_grp_conv_wmma.cpp
    test_instance_string_fwd_grp_conv_dl.cpp
    test_instance_string_bwd_weight_grp_conv_xdl.cpp
    test_instance_string_bwd_weight_grp_conv_dl.cpp
    test_instance_string_bwd_weight_grp_conv_multiple_d_xdl.cpp
    test_instance_string_bwd_weight_grp_conv_two_stage_xdl.cpp
    test_instance_string_bwd_weight_grp_conv_xdl_v3.cpp
)

if (CK_USE_WMMA)
    list(APPEND INSTANCE_STRING_TESTS
        test_instance_string_bwd_weight_grp_conv_wmma_v3.cpp
        test_instance_string_bwd_weight_grp_conv_multiple_d_wmma_v3.cpp
        test_instance_string_bwd_weight_grp_conv_two_stage_wmma_v3.cpp
    )
endif()

add_ck_builder_test(test_ckb_instance_string ${INSTANCE_STRING_TESTS})

# Tests the forward convolution builder across multiple data types and dimensions.
# Individual tests are split into separate files to enable parallel compilation.
add_ck_builder_test(test_ckb_build_fwd_instances
    conv/ck/test_ckb_conv_fwd_2d_bf16_scaleadd_relu.cpp
    conv/ck/test_ckb_conv_fwd_1d_fp16.cpp
    conv/ck/test_ckb_conv_fwd_1d_bf16.cpp
    conv/ck/test_ckb_conv_fwd_1d_i8.cpp
    conv/ck/test_ckb_conv_fwd_2d_fp8.cpp
    conv/ck/test_ckb_conv_fwd_2d_bf16.cpp
    conv/ck/test_ckb_conv_fwd_2d_fp16.cpp
    conv/ck/test_ckb_conv_fwd_2d_fp32.cpp
    conv/ck/test_ckb_conv_fwd_2d_dl_fp16.cpp
    conv/ck/test_ckb_conv_fwd_2d_large_tensor_fp16.cpp
    conv/ck/test_ckb_conv_fwd_3d_bf16.cpp
    conv/ck/test_ckb_conv_fwd_3d_fp16.cpp
    conv/ck/test_ckb_conv_fwd_3d_fp32.cpp
    conv/ck_tile/test_ckb_conv_fwd_2d_fp16_v3.cpp
    conv/ck_tile/test_ckb_conv_bwd_weight_2d_fp16_v3.cpp
    conv/ck_tile/test_ckb_conv_bwd_data_2d_fp16_v3.cpp)
target_link_libraries(test_ckb_build_fwd_instances PRIVATE utility)


################################################################################
# FACTORY TESTS - Expensive Regression Tests (Full MIOpen Kernel Set)
################################################################################
# These tests attempt to build ALL kernels needed by MIOpen for various
# convolution operations. They are extremely expensive (minutes to compile)
# and are intended for deep regression testing and development only.
# NOT suitable for regular CI runs.
#
# Many tests are commented out pending CI support across all GPU architectures.

# Tests the testing utilities themselves
add_ck_factory_test(test_ckb_testing_utils test_testing_utils.cpp)

# TODO: Re-enable these tests once we have CI working across all GPU architectures.
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward test_ck_factory_grouped_convolution_forward.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_clamp test_ck_factory_grouped_convolution_forward_clamp.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_convscale test_ck_factory_grouped_convolution_forward_convscale.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bilinear test_ck_factory_grouped_convolution_forward_bilinear.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scale test_ck_factory_grouped_convolution_forward_scale.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scaleadd_ab test_ck_factory_grouped_convolution_forward_scaleadd_ab.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bias_clamp test_ck_factory_grouped_convolution_forward_bias_clamp.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_bias_bnorm_clamp test_ck_factory_grouped_convolution_forward_bias_bnorm_clamp.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_scaleadd_scaleadd_relu test_ck_factory_grouped_convolution_forward_scaleadd_scaleadd_relu.cpp)
# add_ck_factory_test(test_ckb_factory_grouped_convolution_forward_dynamic_op test_ck_factory_grouped_convolution_forward_dynamic_op.cpp)

################################################################################
# CTest Integration - Register Tests and Assign Labels
################################################################################
# Tests are registered with CTest and labeled for selective execution:
#   - BUILDER_SMOKE: Fast unit tests for frequent development cycles
#   - BUILDER_REGRESSION: Slower integration tests for pre-submission validation

include(CTest)

# Register all smoke tests (fast unit tests, no kernel compilation)
set(CKB_SMOKE_TESTS
    test_ckb_conv_builder
    test_ckb_inline_diff
    test_ckb_conv_traits
    test_ckb_conv_description
    test_ckb_reference_execution
)

foreach(test_target ${CKB_SMOKE_TESTS})
    add_test(NAME ${test_target} COMMAND ${test_target})
    set_tests_properties(${test_target} PROPERTIES LABELS "BUILDER_SMOKE")
endforeach()

# Register all regression tests (integration tests with kernel compilation)
set(CKB_REGRESSION_TESTS
    test_ckb_instance_string
    test_ckb_build_fwd_instances
    test_ckb_testing_utils
    # test_ckb_factory_grouped_convolution_forward_convscale
    # test_ckb_factory_grouped_convolution_forward_scaleadd_ab
    # test_ckb_factory_grouped_convolution_forward_scaleadd_scaleadd_relu
    # test_ckb_factory_grouped_convolution_forward_dynamic_op
)

foreach(test_target ${CKB_REGRESSION_TESTS})
    add_test(NAME ${test_target} COMMAND ${test_target})
    set_tests_properties(${test_target} PROPERTIES LABELS "BUILDER_REGRESSION")
endforeach()

################################################################################
# Custom Build Targets - Convenient Test Execution
################################################################################
# These targets provide convenient ways to build and run different test suites:
#   - smoke-builder:      Quick sanity check during development
#   - regression-builder: Thorough validation before submitting changes
#   - check-builder:      Complete test suite execution

# Helper target to build all smoke tests (without running them)
add_custom_target(build-smoke-builder DEPENDS ${CKB_SMOKE_TESTS})

# Helper target to build all regression tests (without running them)
add_custom_target(build-regression-builder DEPENDS ${CKB_REGRESSION_TESTS})

# Target to run only smoke tests (builds and runs all smoke test executables)
# Use this for quick feedback during active development
add_custom_target(smoke-builder
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "BUILDER_SMOKE"
    DEPENDS build-smoke-builder
    USES_TERMINAL
    COMMENT "Running experimental builder smoke tests..."
)

# Target to run only regression tests (builds and runs all regression test executables)
# Use this before submitting changes to catch integration issues
add_custom_target(regression-builder
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "BUILDER_REGRESSION"
    DEPENDS build-regression-builder
    USES_TERMINAL
    COMMENT "Running experimental builder regression tests..."
)

# Target to run all builder tests (builds and runs all test executables)
# Use this for comprehensive validation
add_custom_target(check-builder
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -R "^test_ckb"
    DEPENDS build-smoke-builder build-regression-builder
    USES_TERMINAL
    COMMENT "Running all experimental builder tests..."
)

################################################################################
# Build Summary
################################################################################

# Print summary of test organization for developer reference
message(STATUS "CK Builder test organization:")
message(STATUS "  Smoke tests: ${CKB_SMOKE_TESTS}")
message(STATUS "  Regression tests: ${CKB_REGRESSION_TESTS}")
