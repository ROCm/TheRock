# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# install(TARGETS) does not install PDBs, so install the comgr PDB explicitly.
if(MSVC AND THEROCK_FLAG_WINDOWS_DRIVER_BUILD AND TARGET amd_comgr)
  get_target_property(_comgr_target_type amd_comgr TYPE)
  if(_comgr_target_type STREQUAL "SHARED_LIBRARY")
    install(
      FILES "$<TARGET_PDB_FILE:amd_comgr>"
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT amd-comgr
      OPTIONAL
    )
  endif()
endif()
