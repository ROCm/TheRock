# therock_setup_linker.cmake
# Configures system LLD for projects not built with amd-llvm toolchain

function(therock_setup_linker _out_lld_path)
  if(WIN32)
    find_program(${_out_lld_path} NAMES lld-link)
  else()
    find_program(${_out_lld_path} NAMES ld.lld lld)
  endif()

  set(${_out_lld_path} "${${_out_lld_path}}" PARENT_SCOPE)
endfunction()
