# Check if a Python executable supports linking against a shared libpython.
# Returns the path to the shared library if available, empty string otherwise.
function(_therock_check_python_shared_library python_exe out_var)
  execute_process(
    COMMAND "${python_exe}" -c "
import sysconfig
import os
# Check if Python was built with --enable-shared
if sysconfig.get_config_var('Py_ENABLE_SHARED'):
    libdir = sysconfig.get_config_var('LIBDIR')
    ldlibrary = sysconfig.get_config_var('LDLIBRARY')
    if libdir and ldlibrary:
        libpath = os.path.join(libdir, ldlibrary)
        if os.path.exists(libpath):
            print(libpath)
            exit(0)
# No shared library available
print('')
"
    OUTPUT_VARIABLE _lib_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _result
  )
  if(_result EQUAL 0 AND _lib_path)
    set(${out_var} "${_lib_path}" PARENT_SCOPE)
  else()
    set(${out_var} "" PARENT_SCOPE)
  endif()
endfunction()

# Detect Python executables that support linking against shared libpython.
# This is required for embedding Python (e.g., in rocgdb).
#
# Logic:
#   1. If THEROCK_SHARED_PYTHON_EXECUTABLES is set, use those but verify each
#      one supports linking against libpython (fatal error if any don't).
#   2. If not set, check if Python3_EXECUTABLE supports linking against
#      libpython and use it if so.
#   3. Otherwise, return an empty list.
#
# The caller should handle an empty list appropriately (e.g., warning and
# disabling Python support).
function(therock_detect_shared_python_executables out_executables)
  set(_result_executables)

  if(THEROCK_SHARED_PYTHON_EXECUTABLES)
    # Make sure our list of python executables does not contain any duplicate
    # entries.
    set(_sanitized_shared_python_executables ${THEROCK_SHARED_PYTHON_EXECUTABLES})
    list(REMOVE_DUPLICATES _sanitized_shared_python_executables)

    message(STATUS "Using explicitly configured shared Python executables: ${_sanitized_shared_python_executables}")
    foreach(_python_exe IN LISTS _sanitized_shared_python_executables)
      if(NOT EXISTS "${_python_exe}")
        message(FATAL_ERROR "Shared Python executable not found: ${_python_exe}")
      endif()
      _therock_check_python_shared_library("${_python_exe}" _lib_path)
      if(NOT _lib_path)
        message(FATAL_ERROR "Python executable does not support shared libpython: ${_python_exe} (from ${_sanitized_shared_python_executables})")
      endif()
      list(APPEND _result_executables "${_python_exe}")
      message(STATUS "  Verified shared libpython at ${_lib_path} for ${_python_exe}")
    endforeach()
  else()
    # Check if the default Python3 has shared library support
    _therock_check_python_shared_library("${Python3_EXECUTABLE}" _lib_path)
    if(_lib_path)
      list(APPEND _result_executables "${Python3_EXECUTABLE}")
      message(STATUS "Default Python supports shared libpython: ${_lib_path}")
    else()
      message(STATUS "Default Python does not support shared libpython (built without --enable-shared)")
    endif()
  endif()

  set(${out_executables} "${_result_executables}" PARENT_SCOPE)
endfunction()

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
