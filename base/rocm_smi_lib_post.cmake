# smi_lib public headers include libdrm, so in the bundled case, we must ensure
# that the installed libraries add the include directory. This is because
# libdrm does not cross the install boundary as a dep (since it is pkgconfig
# and was not set up that way).
# Conversely, if we include directories that do not exist, CMake consumers
# will error out due to the missing directories.
# See the project declaration for where this flag is defined.
if(THEROCK_HAS_BUNDLED_LIBDRM)
  target_include_directories(rocm_smi64 PUBLIC
    "$<INSTALL_INTERFACE:lib/rocm_sysdeps/include>"
  )
  target_include_directories(oam PUBLIC
    "$<INSTALL_INTERFACE:lib/rocm_sysdeps/include>"
  )
endif()
