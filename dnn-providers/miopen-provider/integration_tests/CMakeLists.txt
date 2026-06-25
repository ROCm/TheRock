# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    miopen_plugin_integration_test
    main.cpp
    IntegrationGpuBatchnormBackward.cpp
    IntegrationGpuBatchnormBackwardActivation.cpp
    IntegrationGpuBatchnormForwardInference.cpp
    IntegrationGpuBatchnormFwdPlusActive.cpp
    IntegrationGpuBatchnormForwardTraining.cpp
    IntegrationGpuBatchnormFwdTrainingActiv.cpp
    IntegrationGpuConvForward.cpp
    IntegrationGpuConvFwdBiasActiv.cpp
    IntegrationGpuConvBackwardData.cpp
    IntegrationGpuConvBackwardWeights.cpp
)

target_compile_options(miopen_plugin_integration_test PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# To use the build tree plugin, we use the path relative to the test binary instead of the installed
# libhipdnn_backend.so HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by the sdk
target_compile_definitions(
    miopen_plugin_integration_test
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/miopen_legacy_plugin"
)

target_link_libraries(
    miopen_plugin_integration_test PUBLIC GTest::gtest hip::host hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the miopen_legacy_plugin is built before the miopen_plugin_integration_test, as the
# integration tests depend on the MIOpen legacy plugin
add_dependencies(miopen_plugin_integration_test miopen_legacy_plugin)

if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(miopen_plugin_integration_test)
endif()
add_integration_test_target(miopen_plugin_integration_test ${CMAKE_CURRENT_BINARY_DIR})
