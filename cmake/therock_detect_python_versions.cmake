# Function to detect available Python versions on the system
function(therock_detect_python_versions OUT_VERSIONS)
  cmake_policy(SET CMP0057 NEW)  # Enable IN_LIST operator
  set(_python_versions)
  set(_min_version 8)
  set(_max_version 13)

  # Try to find each Python version
  foreach(_minor RANGE ${_min_version} ${_max_version})
    set(_version "3.${_minor}")

    # Try to find this specific Python version
    find_program(_python_exe
      NAMES python${_version} python${_version}.exe
      PATHS
        /usr/bin
        /usr/local/bin
        /opt/python-${_version}/bin
        $ENV{HOME}/.pyenv/versions/${_version}*/bin
        $ENV{HOME}/.local/bin
        C:/Python${_minor}/
        C:/Python3${_minor}/
      NO_DEFAULT_PATH
    )

    if(_python_exe)
      # Verify the version by running python --version
      execute_process(
        COMMAND ${_python_exe} --version
        OUTPUT_VARIABLE _version_output
        ERROR_VARIABLE _version_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _result
      )

      if(_result EQUAL 0)
        # Extract version from output (Python X.Y.Z)
        if(_version_output MATCHES "Python ${_version}\\.")
          list(APPEND _python_versions ${_version})
          message(STATUS "Found Python ${_version} at ${_python_exe}")
        endif()
      endif()
    endif()

    unset(_python_exe CACHE)
  endforeach()

  # Also check for generic python3 command
  find_program(_python3_exe
    NAMES python3 python3.exe python
    PATHS
      /usr/bin
      /usr/local/bin
      $ENV{HOME}/.local/bin
  )

  if(_python3_exe)
    execute_process(
      COMMAND ${_python3_exe} --version
      OUTPUT_VARIABLE _version_output
      ERROR_VARIABLE _version_error
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _result
    )

    if(_result EQUAL 0 AND _version_output MATCHES "Python 3\\.([0-9]+)\\.")
      set(_minor "${CMAKE_MATCH_1}")
      if(_minor GREATER_EQUAL _min_version AND _minor LESS_EQUAL _max_version)
        set(_version "3.${_minor}")
        if(NOT _version IN_LIST _python_versions)
          list(APPEND _python_versions ${_version})
          message(STATUS "Found Python ${_version} at ${_python3_exe}")
        endif()
      endif()
    endif()
  endif()

  # Sort the versions
  if(_python_versions)
    list(SORT _python_versions)
    list(REMOVE_DUPLICATES _python_versions)
  else()
    message(WARNING "No Python versions between 3.${_min_version} and 3.${_max_version} found on the system")
  endif()

  set(${OUT_VERSIONS} "${_python_versions}" PARENT_SCOPE)
endfunction()
