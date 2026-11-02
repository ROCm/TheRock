# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Dyninst and rocprofiler-systems internal libraries live in lib/rocprofiler-systems.
# Set the RPATH origin per-target so that (for example) lib/rocm_sysdeps/lib resolves as
# $ORIGIN/../rocm_sysdeps/lib rather than $ORIGIN/rocm_sysdeps/lib.
set(_rocprofsys_lib_targets
  common
  dynElf
  dynDwarf
  dyninstAPI
  dyninstAPI_RT
  dynC_API
  instructionAPI
  parseAPI
  patchAPI
  pcontrol
  stackwalk
  symtabAPI
  gotcha
)

foreach(_target ${_rocprofsys_lib_targets})
  if(TARGET "${_target}")
    set_target_properties("${_target}" PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN lib/rocprofiler-systems
    )
  endif()
endforeach()

# timem explicitly disables PIE because it links static PAPI libraries that
# are not position independent. Preserve that upstream requirement.
foreach(_target timem timem-mpi)
  if(TARGET "${_target}")
    set_property(TARGET "${_target}" PROPERTY THEROCK_DISABLE_HARDENING ON)
  endif()
endforeach()
