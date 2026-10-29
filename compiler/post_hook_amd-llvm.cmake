# post_hook_amd-llvm.cmake (runs during amd-llvm configure; rules run at install time)
# With the introduction of LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON these libraries are
# installed one level down and could break backwards compatability with older ROCm
# versions. For now we will install symlinks in the old lib/llvm/lib directory.
#
# Add symlink for x86_64-pc-linux-gnu -> x86_64-unknown-linux-gnu to ensure
# projects that hardcode -target will still work.

#TODO: Remove entire post_hook for next major ROCm version after 10.0
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/GetHostTriple.cmake")
get_host_triple(THEROCK_LLVM_HOST_TRIPLE)

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules/LLVMVersion.cmake")
  if(NOT LLVM_VERSION_MAJOR)
    message(FATAL_ERROR "LLVM_VERSION_MAJOR is empty; cannot create clang resource-dir triple symlink")
  endif()
  if(THEROCK_LLVM_HOST_TRIPLE MATCHES "-unknown-")
    string(REPLACE "-unknown-" "-pc-" _pc_triple "${THEROCK_LLVM_HOST_TRIPLE}")
    install(CODE "
      set(_clang_lib \"\${CMAKE_INSTALL_PREFIX}/lib/clang/${LLVM_VERSION_MAJOR}/lib\")
      set(_src \"\${_clang_lib}/${THEROCK_LLVM_HOST_TRIPLE}\")
      set(_dest \"\${_clang_lib}/${_pc_triple}\")
      if(EXISTS \"\${_src}\" AND NOT EXISTS \"\${_dest}\")
        file(CREATE_LINK \"${THEROCK_LLVM_HOST_TRIPLE}\" \"\${_dest}\" SYMBOLIC)
        message(STATUS \"Created triple compat symlink: \${_dest} -> ${THEROCK_LLVM_HOST_TRIPLE}\")
      endif()
    ")
  endif()

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

    # compiler-rt sanitizers: per-target dir is
    #   lib/clang/<maj>/lib/<triple>/libclang_rt.*.so
    # Legacy layout was lib/clang/<maj>/lib/linux/libclang_rt.*-<arch>.so.
    # Symlink those arch-suffixed names to the per-target unsuffixed files.
    string(REGEX REPLACE "^([^-]+).*" "\\1" _clang_rt_arch "${THEROCK_LLVM_HOST_TRIPLE}")
    set(_compat_clang_rt_libs
      libclang_rt.asan.so
      libclang_rt.dyndd.so
      libclang_rt.hwasan_aliases.so
      libclang_rt.hwasan.so
      libclang_rt.memprof.so
      libclang_rt.nsan.so
      libclang_rt.scudo_standalone.so
      libclang_rt.tsan.so
      libclang_rt.ubsan_minimal.so
      libclang_rt.ubsan_standalone.so
    )
    foreach(_lib IN LISTS _compat_clang_rt_libs)
      # libclang_rt.asan.so -> libclang_rt.asan-x86_64.so
      string(REGEX REPLACE "\\.so$" "-${_clang_rt_arch}.so" _legacy_lib "${_lib}")
      install(CODE "
        set(_clang_lib \"\${CMAKE_INSTALL_PREFIX}/lib/clang/${LLVM_VERSION_MAJOR}/lib\")
        set(_src \"\${_clang_lib}/${THEROCK_LLVM_HOST_TRIPLE}/${_lib}\")
        set(_dest_dir \"\${_clang_lib}/linux\")
        set(_dest \"\${_dest_dir}/${_legacy_lib}\")
        if(EXISTS \"\${_src}\")
          file(MAKE_DIRECTORY \"\${_dest_dir}\")
          if(NOT EXISTS \"\${_dest}\")
            file(CREATE_LINK \"../${THEROCK_LLVM_HOST_TRIPLE}/${_lib}\" \"\${_dest}\" SYMBOLIC)
            message(STATUS \"Created clang_rt compat symlink: \${_dest} -> ../${THEROCK_LLVM_HOST_TRIPLE}/${_lib}\")
          endif()
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
