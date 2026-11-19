# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The public HSA layout is include/hsa/*.h. Some source headers still include
# siblings through ROCr's source-relative inc/ spelling. Resolve those includes
# in a generated copy so installed consumers need neither a private directory
# nor AMD_INTERNAL_BUILD. Keep every dependent public header from the same tree.
function(rocm_runtimes_prepare_hsa_headers source destination)
  file(GLOB _previous_headers "${destination}/*.h")
  foreach(_header IN LISTS _previous_headers)
    cmake_path(GET _header FILENAME _name)
    if(NOT EXISTS "${source}/${_name}")
      file(REMOVE "${_header}")
    endif()
  endforeach()
  file(GLOB _headers CONFIGURE_DEPENDS "${source}/*.h")
  foreach(_header IN LISTS _headers)
    cmake_path(GET _header FILENAME _name)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_header}")
    file(READ "${_header}" _contents)
    string(REGEX REPLACE "(#[ \t]*include[ \t]*\")inc/" "\\1" _contents "${_contents}")
    set(_output "${destination}/${_name}")
    if(EXISTS "${_output}")
      file(READ "${_output}" _previous)
      if(_previous STREQUAL _contents)
        continue()
      endif()
    endif()
    file(WRITE "${_output}" "${_contents}")
  endforeach()
endfunction()
