# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14)
project(TheRockGTestSamples CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()
include(CTest)

# Find Google Test
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    
    # Prevent overriding the parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Compiler options
if(MSVC)
    add_compile_options(/W4 /WX-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Helper function to add a test executable
function(add_gtest_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    
    target_link_libraries(${target_name}
        PRIVATE
            GTest::gtest
            GTest::gtest_main
    )
    
    # Add as CTest
    add_test(NAME ${target_name} COMMAND ${target_name})
    
    # Set test properties
    set_tests_properties(${target_name} PROPERTIES
        TIMEOUT 60
        LABELS "gtest;samples"
    )
    
    # Create output directory for test results
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endfunction()

# Add test executables
add_gtest_executable(sample_math_tests sample_math_tests.cpp)
add_gtest_executable(sample_string_tests sample_string_tests.cpp)
add_gtest_executable(sample_integration_tests sample_integration_tests.cpp)

# Add custom target to run all GTest samples
add_custom_target(run_gtest_samples
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L gtest
    DEPENDS sample_math_tests sample_string_tests sample_integration_tests
    COMMENT "Running all GTest samples"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Add custom target to generate test reports
add_custom_target(gtest_reports
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test_results
    COMMAND $<TARGET_FILE:sample_math_tests> --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results/math_tests.xml
    COMMAND $<TARGET_FILE:sample_string_tests> --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results/string_tests.xml
    COMMAND $<TARGET_FILE:sample_integration_tests> --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results/integration_tests.xml
    DEPENDS sample_math_tests sample_string_tests sample_integration_tests
    COMMENT "Generating GTest XML reports"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Install test executables (optional)
install(TARGETS sample_math_tests sample_string_tests sample_integration_tests
    RUNTIME DESTINATION tests
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== TheRock GTest Samples Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Test executables:")
message(STATUS "  - sample_math_tests")
message(STATUS "  - sample_string_tests")
message(STATUS "  - sample_integration_tests")
message(STATUS "==========================================")
message(STATUS "")

