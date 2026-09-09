# post_hook_amd-llvm.cmake (runs during amd-llvm configure; rules run at install time)
# With the introduction of LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON these libraries are
# installed one level down and could break backwards compatability with older ROCm
# versions. For now we will install symlinks in the old lib/llvm/lib directory.

#TODO: Remove for next major ROCm version after 10.0
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/GetHostTriple.cmake")
get_host_triple(THEROCK_LLVM_HOST_TRIPLE)

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  if(THEROCK_LLVM_HOST_TRIPLE)
    set(_compat_libs
      libarcher.so
      libc++.so
      libc++abi.so
      libiomp5.so
      libgomp.so
      libgomp.so.1
      libomp.so
      libomptarget.so
      libomptest.so
      libLLVMOffload.so
      libLLVMOffloadKernel.so
      libunwind.so
    )
    foreach(_lib IN LISTS _compat_libs)
      install(CODE "
        set(_dest \"\${CMAKE_INSTALL_PREFIX}/lib/${_lib}\")
        set(_src \"\${CMAKE_INSTALL_PREFIX}/lib/${THEROCK_LLVM_HOST_TRIPLE}/${_lib}\")
        if(EXISTS \"\${_src}\" AND NOT EXISTS \"\${_dest}\")
          file(CREATE_LINK \"${THEROCK_LLVM_HOST_TRIPLE}/${_lib}\" \"\${_dest}\" SYMBOLIC)
          message(STATUS \"Created symlink: \${_dest} -> ${THEROCK_LLVM_HOST_TRIPLE}/${_lib}\")
        endif()
      ")
    endforeach()
  else()
    message(FATAL_ERROR "THEROCK_LLVM_HOST_TRIPLE not found, but LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON.
      The backwards compatible symlinks are required until ROCm 11.0.")
  endif()
else()
  message(STATUS "LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF, skipping backwards compatible symlinks.")
endif()
