# This variable allows building for multiple Python versions by specifying their executables.
# Note: Most projects do not need to set this; only use it for multi-version Python builds.
# Usage scenarios:
#   a. Defined Python3 Executables: an explicit list of python interpreters to build for
#      Example: -DTHEROCK_DIST_PYTHON_EXECUTABLES="/opt/python-3.8/bin/python3.8;/opt/python-3.9/bin/python3.9"
#   b. Default Python3 Available: only build for the single auto detected python version (default behavior)
#
# For manylinux builds, this should be set to a subset of Python versions from /opt/python-*/bin
# For regular builds, if not set, it defaults to the system Python3_EXECUTABLE

function(therock_detect_python_versions OUT_EXECUTABLES OUT_VERSIONS)
  set(_python_executables)
  set(_python_versions)

  if(THEROCK_DIST_PYTHON_EXECUTABLES)
    # Use the explicitly provided list of Python executables
    message(STATUS "Using explicitly configured Python executables: ${THEROCK_DIST_PYTHON_EXECUTABLES}")

    foreach(_python_exe IN LISTS THEROCK_DIST_PYTHON_EXECUTABLES)
      if(EXISTS "${_python_exe}")
        # Verify this is actually a Python executable and get its version
        execute_process(
          COMMAND "${_python_exe}" --version
          OUTPUT_VARIABLE _version_output
          ERROR_VARIABLE _version_error
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_STRIP_TRAILING_WHITESPACE
          RESULT_VARIABLE _result
        )

        if(_result EQUAL 0 AND _version_output MATCHES "Python ([0-9]+)\\.([0-9]+)\\.")
          set(_major "${CMAKE_MATCH_1}")
          set(_minor "${CMAKE_MATCH_2}")
          set(_version "${_major}.${_minor}")

          list(APPEND _python_executables "${_python_exe}")
          list(APPEND _python_versions "${_version}")
          message(STATUS "  Verified Python ${_version} at ${_python_exe}")
        else()
          message(FATAL_ERROR "  Failed to verify Python at ${_python_exe}")
        endif()
      else()
        message(FATAL_ERROR "  Python executable not found: ${_python_exe}")
      endif()
    endforeach()
  else()
    # Default behavior: find and use only the system Python
    find_package(Python3 COMPONENTS Interpreter)

    if(Python3_FOUND)
      list(APPEND _python_executables "${Python3_EXECUTABLE}")
      list(APPEND _python_versions "${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
      message(STATUS "Using system Python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR} at ${Python3_EXECUTABLE}")
    else()
      message(FATAL_ERROR "No Python 3 interpreter found on the system")
    endif()
  endif()

  # Set output variables
  set("${OUT_EXECUTABLES}" "${_python_executables}" PARENT_SCOPE)
  set("${OUT_VERSIONS}" "${_python_versions}" PARENT_SCOPE)

  if(NOT _python_executables)
    message(FATAL_ERROR "No Python executables configured or found")
  endif()
endfunction()
