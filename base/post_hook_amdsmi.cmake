# Test binary installed to share/amd_smi/tests/, but libamd_smi.so is under lib/
if(TARGET amdsmitst)
  set_target_properties(amdsmitst PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "share/amd_smi/tests")
  # Defer prepending $ORIGIN until after TheRock sets INSTALL_RPATH. Why? Because otherwise it will be relative.
  # This is needed because amdsmi builds its own gtest and installs it under share/amd_smi/tests/. It can be removed if amdsmi starts using TheRock's gtest instead.
  cmake_language(DEFER CALL _amdsmi_prepend_origin_to_rpath)
endif()

function(_amdsmi_prepend_origin_to_rpath)
  if(TARGET amdsmitst)
    get_target_property(_rpath amdsmitst INSTALL_RPATH)
    if(_rpath)
      set_target_properties(amdsmitst PROPERTIES INSTALL_RPATH "$ORIGIN;${_rpath}")
    endif()
  endif()
endfunction()
