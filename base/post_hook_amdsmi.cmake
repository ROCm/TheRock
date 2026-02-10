# Test binary installed to share/amd_smi/tests/, but libamd_smi.so is under lib/
if(TARGET amdsmitst)
  set_target_properties(amdsmitst PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "share/amd_smi/tests")
endif()
