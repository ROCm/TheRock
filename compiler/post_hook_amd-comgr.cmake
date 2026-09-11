# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install the linker generated PDB beside the comgr DLL.
#
# install(TARGETS) does not install PDB files, so without this the PDB is
# produced in the build tree and then discarded at stage time. Scoped to the
# Windows driver build: that is the configuration whose DLL is submitted to the
# MSFT Hardware Dev Center for WHQL signing, where the symbols are stripped and
# indexed for crash telemetry. The PDB is not shipped to end users; it is routed
# into the dbg artifact component (see artifact-amd-llvm.toml).
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
