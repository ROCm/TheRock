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
  if(NOT THEROCK_LLVM_HOST_TRIPLE)
    message(FATAL_ERROR "THEROCK_LLVM_HOST_TRIPLE not found, but LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON.
      The backwards compatible symlinks are required until ROCm 11.0.")
  endif()
  # Distribution installs skip install(CODE). The stage step runs this same file.
  set(_compat_script "${CMAKE_CURRENT_LIST_DIR}/../cmake/therock_amd_llvm_per_target_compat_symlinks.cmake")
  install(CODE "
    set(PREFIX \"\${CMAKE_INSTALL_PREFIX}\")
    set(HOST_TRIPLE \"${THEROCK_LLVM_HOST_TRIPLE}\")
    set(LLVM_VERSION_MAJOR \"${LLVM_VERSION_MAJOR}\")
    include(\"${_compat_script}\")
  ")
else()
  message(STATUS "LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF, skipping backwards compatible symlinks.")
endif()
