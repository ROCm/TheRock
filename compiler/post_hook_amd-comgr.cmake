# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TheRock always passes -DCOMGR_STATIC_LLVM=ON. If that did not stick, or if
# comgr still resolved the dylib branch, libamd_comgr would DT_NEEDED libLLVM
# and publish the LLVM C API into RTLD_DEFAULT under RTLD_GLOBAL HIP preload
# (the 10.0.0 failure mode).
#
# Assert the chosen link line, not LLVM_LINK_LLVM_DYLIB / CLANG_LINK_CLANG_DYLIB:
# comgr sets those OFF unconditionally under COMGR_STATIC_LLVM, so a flag
# assertion is unreachable. LLVM_LIBS / CLANG_LIBS are what the link uses.
# Linux-only: Windows already builds LLVM without LLVM_LINK_LLVM_DYLIB.
if(NOT WIN32)
  if(NOT COMGR_STATIC_LLVM)
    message(FATAL_ERROR
      "TheRock requires COMGR_STATIC_LLVM=ON for amd-comgr. "
      "libamd_comgr would DT_NEEDED libLLVM and publish LLVM C API "
      "into RTLD_DEFAULT under RTLD_GLOBAL HIP preload.")
  endif()
  if("LLVM" IN_LIST LLVM_LIBS)
    message(FATAL_ERROR
      "COMGR_STATIC_LLVM=ON but LLVM_LIBS still contains the 'LLVM' dylib "
      "target: ${LLVM_LIBS}. libamd_comgr would DT_NEEDED libLLVM. "
      "Expected the llvm_map_components_to_libnames static list.")
  endif()
  if("clang-cpp" IN_LIST CLANG_LIBS)
    message(FATAL_ERROR
      "COMGR_STATIC_LLVM=ON but CLANG_LIBS still contains 'clang-cpp': "
      "${CLANG_LIBS}. libamd_comgr would DT_NEEDED libclang-cpp. "
      "Expected clangBasic/clangDriver/... static libs.")
  endif()
endif()
