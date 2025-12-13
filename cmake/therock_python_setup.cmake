# therock_python_setup.cmake
# Python initialization and build topology generation.
# Python is used throughout the build system for various tasks including
# generating build topology targets from BUILD_TOPOLOGY.toml.

# Function: therock_setup_python_and_topology
# Initializes Python interpreter and generates build topology CMake targets.
#
# This function:
#   1. Finds Python 3.9+ (required for the build)
#   2. Sets up dependency tracking for BUILD_TOPOLOGY.toml
#   3. Generates therock_topology.cmake from BUILD_TOPOLOGY.toml
#
# The generated topology file contains feature definitions for artifacts
# defined in BUILD_TOPOLOGY.toml.
#
function(therock_setup_python_and_topology)
  ################################################################################
  # Python Initialization
  ################################################################################
  # Python is used throughout the build. Ensure it is initialized early.

  find_package(Python3 3.9 COMPONENTS Interpreter REQUIRED)

  ################################################################################
  # Generate Build Topology Targets
  ################################################################################
  # Track BUILD_TOPOLOGY.toml and related files for changes to trigger reconfigure
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/BUILD_TOPOLOGY.toml"
    "${CMAKE_CURRENT_SOURCE_DIR}/build_tools/topology_to_cmake.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/build_tools/_therock_utils/build_topology.py"
  )

  block(SCOPE_FOR VARIABLES)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cmake")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/build_tools/topology_to_cmake.py"
        "--topology" "${CMAKE_CURRENT_SOURCE_DIR}/BUILD_TOPOLOGY.toml"
        "--output" "${CMAKE_CURRENT_BINARY_DIR}/cmake/therock_topology.cmake"
      OUTPUT_VARIABLE _topology_output
      ERROR_VARIABLE _topology_error
      COMMAND_ERROR_IS_FATAL ANY  # Fail on any error
    )
    message(STATUS "Generated build topology targets from BUILD_TOPOLOGY.toml")
  endblock()
endfunction()
