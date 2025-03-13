target_include_directories(rocm_smi64 PUBLIC
  "$<INSTALL_INTERFACE:lib/rocm_sysdeps/include>"
)
target_include_directories(oam PUBLIC
  "$<INSTALL_INTERFACE:lib/rocm_sysdeps/include>"
)
