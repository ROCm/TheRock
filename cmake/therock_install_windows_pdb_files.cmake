# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This script is included during cmake_install.cmake at the top level of any
# sub-project which is collecting linker PDB files into debug artifacts on
# Windows.
# It runs with the following variables in scope:
#   THEROCK_DEBUG_PDB_RECORDS: List of records in the form "target|pdb_path".
#   THEROCK_DEBUG_PDB_SUBDIR: Stable sub-directory for this sub-project.
#   THEROCK_STAGE_INSTALL_ROOT: The root of the stage installation tree.

block(SCOPE_FOR VARIABLES)
  foreach(record ${THEROCK_DEBUG_PDB_RECORDS})
    string(REGEX MATCH "^([^|]+)\\|(.*)$" _record_match "${record}")
    if(NOT _record_match)
      message(WARNING "Skipping malformed PDB record: ${record}")
      continue()
    endif()

    set(_target_name "${CMAKE_MATCH_1}")
    set(_pdb_path "${CMAKE_MATCH_2}")
    if(NOT EXISTS "${_pdb_path}")
      message(VERBOSE "Skipping PDB for ${_target_name} (${_pdb_path} does not exist)")
      continue()
    endif()

    cmake_path(GET _pdb_path FILENAME _pdb_name)
    set(_output_path "${THEROCK_STAGE_INSTALL_ROOT}/.debug/pdb/${THEROCK_DEBUG_PDB_SUBDIR}/${_target_name}/${_pdb_name}")
    cmake_path(GET _output_path PARENT_PATH _parent_dir)
    file(MAKE_DIRECTORY "${_parent_dir}")
    message(STATUS "Installing PDB from ${_pdb_path} to ${_output_path}")
    file(COPY_FILE "${_pdb_path}" "${_output_path}" ONLY_IF_DIFFERENT)
  endforeach()
endblock()