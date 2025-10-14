# Enables to build a subcomponent from a user provided source location.
function(therock_enable_external_source package_name default_source_dir option_default)
  string(TOUPPER ${package_name} _PACKAGE_UPPER)
  string(REPLACE "-" "_" _PACKAGE ${_PACKAGE_UPPER})
  option(THEROCK_USE_EXTERNAL_${_PACKAGE} "Use external ${package_name} source location" ${option_default})

  if(THEROCK_USE_EXTERNAL_${_PACKAGE})
    # If the source should come from an external package, the source dir must be
    # set manually.
    if(NOT THEROCK_${_PACKAGE}_SOURCE_DIR)
        message(FATAL_ERROR "If THEROCK_USE_EXTERNAL_${_PACKAGE} is set, THEROCK_${_PACKAGE}_SOURCE_DIR is required!")
    endif()
    cmake_path(ABSOLUTE_PATH THEROCK_${_PACKAGE}_SOURCE_DIR NORMALIZE OUTPUT_VARIABLE _source_dir)

    # Check if the user provided source directory exists.
    if(NOT EXISTS ${_source_dir})
        message(FATAL_ERROR "THEROCK_${_PACKAGE}_SOURCE_DIR points to '${_source_dir}' which does not exist!")
    endif()
    set(THEROCK_${_PACKAGE}_SOURCE_DIR "${_source_dir}" CACHE STRING "Path to ${package_name} sources" FORCE)
  else()
    set(THEROCK_${_PACKAGE}_SOURCE_DIR "${default_source_dir}" CACHE STRING "Path to ${package_name} sources" FORCE)
  endif()
endfunction()
