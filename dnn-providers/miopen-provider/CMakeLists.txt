# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
    message(STATUS "Building miopen legacy plugin standalone")
    cmake_minimum_required(VERSION 3.25.2)
    add_compile_definitions(__HIP_PLATFORM_AMD__)

    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/cmake/ClangToolChain.cmake"
        CACHE STRING "Toolchain file"
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    set(MIOPEN_LEGACY_PLUGIN "0.0.1")
    project(hipDNN VERSION ${MIOPEN_LEGACY_PLUGIN} LANGUAGES CXX)

    if(NOT WIN32)
        # Only set if not already specified by user
        if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
            set(CMAKE_INSTALL_PREFIX "${ROCM_PATH}" CACHE PATH "Install path prefix" FORCE)
        endif()
        message(STATUS "Setting CMAKE_INSTALL_PREFIX to ${CMAKE_INSTALL_PREFIX}")
    endif()

    include(GNUInstallDirs)

    set(CMAKE_CXX_STANDARD 17)

    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE
        CACHE BOOL "Add paths to linker search and installed rpath"
    )

    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${ROCM_PATH}/lib/cmake/hip
         ${HIP_DIR}/cmake
    )

    # Enable testing for standalone build
    # Migrate HIP_DNN_SKIP_TESTS -> HIPDNN_SKIP_TESTS
    if(DEFINED HIP_DNN_SKIP_TESTS)
        set(HIPDNN_SKIP_TESTS ${HIP_DNN_SKIP_TESTS} CACHE BOOL "Skip building tests" FORCE)
        unset(HIP_DNN_SKIP_TESTS CACHE)
        message(DEPRECATION "HIP_DNN_SKIP_TESTS has been renamed to HIPDNN_SKIP_TESTS. Please update your build scripts to use -DHIPDNN_SKIP_TESTS instead.")
    endif()
    option(HIPDNN_SKIP_TESTS "Skip building tests" OFF)

    list(
        PREPEND
        HIPDNN_WARNING_COMPILE_OPTIONS
        -Werror # Treat all warnings as errors
        -Wall # Enable most common warnings
        -Wextra # Enable additional warnings not covered by -Wall
        -Wpedantic # Enforce strict ISO C++ compliance
        -Wshadow # Warn about variable shadowing
        -Wnon-virtual-dtor # Warn if a class with virtual functions has a non-virtual destructor
        -Wold-style-cast # Warn about C-style casts
        -Wcast-align # Warn about potential performance issues with misaligned casts
        -Woverloaded-virtual # Warn if a base class function is hidden by a derived class function
                             # with the same name
        -Wconversion # Warn about implicit type conversions that may alter a value
        -Wsign-conversion # Warn about implicit conversions between signed and unsigned types
        -Wnull-dereference # Warn about dereferencing null pointers
        -Wdouble-promotion # Warn when a float is implicitly promoted to a double
        -Wformat=2 # Enable stricter format string checks
        -Winit-self # Warn about variables initialized with itself
        -Wunreachable-code # Warn about unreachable code
        -Wno-error=unused-command-line-argument # Disable error for unused command line arguments
        -Wswitch-default # Warn if a switch statement does not have a default case
    )

    # NOTE: Workaround until llvm & hip cmake modules fixes symlink logic in their config files;
    # remove when fixed. Added to prevent issues with not finding hip cmake.
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip)

    find_package(hip REQUIRED)
    find_package(hipdnn_data_sdk CONFIG REQUIRED)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)

    # Setup testing dependencies & tests for standalone build
    include(Dependencies)
    include(Tests)

else()
    message(STATUS "Building miopen legacy plugin as a dependency")
endif()

if(WIN32)
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "bin")
else()
    set(HIPDNN_PLUGIN_ENGINE_ROOTDIR "lib")
endif()

set(HIPDNN_BUILD_PLUGIN_ENGINE_DIR
    "${CMAKE_BINARY_DIR}/${HIPDNN_PLUGIN_ENGINE_ROOTDIR}/${HIPDNN_PLUGIN_ENGINE_SUBDIR}"
)

message(STATUS "Plugin build directory: ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}")
message(STATUS "Plugin install directory: ${HIPDNN_INSTALL_PLUGIN_ENGINE_DIR}")
file(MAKE_DIRECTORY ${HIPDNN_BUILD_PLUGIN_ENGINE_DIR})

find_package(miopen CONFIG REQUIRED)

add_library(
    miopen_legacy_plugin_private
    engines/MiopenEngine.cpp
    engines/plans/MiopenBatchnormApplicabilityChecks.cpp
    engines/plans/MiopenBatchnormPlanBuilder.cpp
    engines/plans/MiopenBatchnormBwdPlan.cpp
    engines/plans/MiopenBatchnormFwdInferencePlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlan.cpp
    engines/plans/MiopenBatchnormFwdTrainingPlanBuilder.cpp
    engines/plans/MiopenConvBwdPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlan.cpp
    engines/plans/MiopenConvFwdBiasActivPlanBuilder.cpp
    engines/plans/MiopenConvFwdPlan.cpp
    engines/plans/MiopenConvPlanBuilder.cpp
    engines/plans/MiopenConvWrwPlan.cpp
    EngineManager.cpp
    MiopenActivationDescriptor.cpp
    MiopenHandleFactory.cpp
    MiopenContainer.cpp
    MiopenConvDescriptor.cpp
    MiopenUtils.cpp
    MiopenTensor.cpp
    MiopenLegacyPlugin.cpp
)
target_compile_definitions(
    miopen_legacy_plugin_private PRIVATE COMPONENT_NAME="miopen_legacy_plugin"
)
target_include_directories(miopen_legacy_plugin_private PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(miopen_legacy_plugin_private PUBLIC MIOpen hipdnn_data_sdk hipdnn_plugin_sdk)
target_compile_options(miopen_legacy_plugin_private PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(miopen_legacy_plugin_private)
endif()

add_library(miopen_legacy_plugin SHARED MiopenLegacyPluginPublic.cpp)
target_compile_definitions(miopen_legacy_plugin PRIVATE COMPONENT_NAME="miopen_legacy_plugin")
target_link_libraries(miopen_legacy_plugin PRIVATE miopen_legacy_plugin_private)
target_link_libraries(
    miopen_legacy_plugin PRIVATE "-Xlinker --exclude-libs=ALL"
) # Hide symbols from linked static libs
target_compile_options(miopen_legacy_plugin PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})
set_target_properties(
    miopen_legacy_plugin
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               LIBRARY_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}"
               INSTALL_RPATH "${ROCM_PATH}/lib"
               INSTALL_RPATH_USE_LINK_PATH TRUE
               BUILD_WITH_INSTALL_RPATH TRUE
)

if(BUILD_PLUGIN_AS_DEPENDENCY)
    clang_tidy_check(miopen_legacy_plugin)
endif()

if(NOT HIPDNN_SKIP_TESTS)
    add_subdirectory(tests)
    add_subdirectory(integration_tests)

    if(NOT BUILD_PLUGIN_AS_DEPENDENCY)
        # Keep this after all build folders have been added so they have a chance to register their tests
        # using add_*_test_target()
        finalize_test_targets()

        # For standalone builds we need to install the ctest files
        install_miopen_legacy_plugin_ctest_files()
    endif()
endif()

install(TARGETS miopen_legacy_plugin EXPORT miopen-legacy-plugin
        LIBRARY DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
        RUNTIME DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}
)
