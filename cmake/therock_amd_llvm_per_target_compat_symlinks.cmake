# Per-target runtime compat symlinks for a staged LLVM prefix.
#   install(CODE) include() from compiler/post_hook_amd-llvm.cmake
#   cmake -P after an amd-llvm distribution stage
# Requires PREFIX and HOST_TRIPLE. LLVM_VERSION_MAJOR is optional.
if(NOT PREFIX OR NOT HOST_TRIPLE)
  message(FATAL_ERROR "PREFIX and HOST_TRIPLE are required")
endif()

if(NOT LLVM_VERSION_MAJOR)
  file(GLOB _clang_ver_dirs LIST_DIRECTORIES true "${PREFIX}/lib/clang/*")
  if(_clang_ver_dirs)
    list(SORT _clang_ver_dirs)
    list(GET _clang_ver_dirs -1 _clang_ver_dir)
    get_filename_component(LLVM_VERSION_MAJOR "${_clang_ver_dir}" NAME)
  endif()
endif()

if(LLVM_VERSION_MAJOR AND HOST_TRIPLE MATCHES "-unknown-")
  string(REPLACE "-unknown-" "-pc-" _pc_triple "${HOST_TRIPLE}")
  set(_clang_lib "${PREFIX}/lib/clang/${LLVM_VERSION_MAJOR}/lib")
  set(_src "${_clang_lib}/${HOST_TRIPLE}")
  set(_dest "${_clang_lib}/${_pc_triple}")
  if(EXISTS "${_src}" AND NOT EXISTS "${_dest}")
    file(CREATE_LINK "${HOST_TRIPLE}" "${_dest}" SYMBOLIC)
    message(STATUS "Created triple compat symlink: ${_dest} -> ${HOST_TRIPLE}")
  endif()
endif()

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
  set(_dest "${PREFIX}/lib/${_lib}")
  set(_src "${PREFIX}/lib/${HOST_TRIPLE}/${_lib}")
  if(EXISTS "${_src}" AND NOT EXISTS "${_dest}")
    file(CREATE_LINK "${HOST_TRIPLE}/${_lib}" "${_dest}" SYMBOLIC)
    message(STATUS "Created symlink: ${_dest} -> ${HOST_TRIPLE}/${_lib}")
  endif()
endforeach()

# compiler-rt sanitizers: per-target dir is
#   lib/clang/<maj>/lib/<triple>/libclang_rt.*.so
# Legacy layout was lib/clang/<maj>/lib/linux/libclang_rt.*-<arch>.so.
if(LLVM_VERSION_MAJOR)
  string(REGEX REPLACE "^([^-]+).*" "\\1" _clang_rt_arch "${HOST_TRIPLE}")
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
    string(REGEX REPLACE "\\.so$" "-${_clang_rt_arch}.so" _legacy_lib "${_lib}")
    set(_clang_lib "${PREFIX}/lib/clang/${LLVM_VERSION_MAJOR}/lib")
    set(_src "${_clang_lib}/${HOST_TRIPLE}/${_lib}")
    set(_dest_dir "${_clang_lib}/linux")
    set(_dest "${_dest_dir}/${_legacy_lib}")
    if(EXISTS "${_src}")
      file(MAKE_DIRECTORY "${_dest_dir}")
      if(NOT EXISTS "${_dest}")
        file(CREATE_LINK "../${HOST_TRIPLE}/${_lib}" "${_dest}" SYMBOLIC)
        message(STATUS "Created clang_rt compat symlink: ${_dest} -> ../${HOST_TRIPLE}/${_lib}")
      endif()
    endif()
  endforeach()
endif()
