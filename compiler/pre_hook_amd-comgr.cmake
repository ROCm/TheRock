if(WIN32)
  set(LLVM_LINK_LLVM_DYLIB OFF)
else()
  set(LLVM_LINK_LLVM_DYLIB ON)
endif()

set(COMGR_DISABLE_SPIRV OFF)

# The comgr tests have a circular dependency on the HIP runtime.
# https://github.com/ROCm/TheRock/issues/67
set(BUILD_TESTING OFF CACHE BOOL "DISABLE BUILDING TESTS IN SUBPROJECTS" FORCE)

set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/llvm/lib;$ORIGIN/rocm_sysdeps/lib")

# See Comgr::LoadLib in clr comgrctx.cpp: On windows, this expects the cmgr
# library to have a versioned output name, but there does not seem to be a
# public patch to llvm-project/amd/comgr which sets it properly. Therefore,
# we align it with what LoadLib expects:
#   amd_comgrMMNN.dll
# where MM is the left zero padded HIP_MAJOR_VERSION and NN is the left zero
# padded HIP_MINOR_VERSION. There is no natural connection between these
# projects in the codebase, so this is just an action at a distance / must
# line up thing. We require the version explicitly from the caller and complain
# loudly if not present.
function(therock_patch_comgr_win_output_name)
  if(NOT DEFINED THEROCK_HIP_MAJOR_VERSION OR NOT DEFINED THEROCK_HIP_MINOR_VERSION)
    message(FATAL_ERROR "Super-project must set THEROCK_HIP_MAJOR_VERSION and THEROCK_HIP_MINOR_VERSION")
  endif()
  if(THEROCK_HIP_MAJOR_VERSION LESS_EQUAL 9)
    set(THEROCK_HIP_MAJOR_VERSION "0${THEROCK_HIP_MAJOR_VERSION}")
  endif()
  if(THEROCK_HIP_MINOR_VERSION LESS_EQUAL 9)
    set(THEROCK_HIP_MINOR_VERSION "0${THEROCK_HIP_MINOR_VERSION}")
  endif()

  set(suffix_version "${THEROCK_HIP_MAJOR_VERSION}${THEROCK_HIP_MINOR_VERSION}")
  message(STATUS "Versioned comgr suffix: ${suffix_version}")

  if(WIN32)
    set(output_name "amd_comgr${suffix_version}")
    message(STATUS "Override comgr output_name (windows): ${output_name}")
    set_target_properties(amd_comgr PROPERTIES OUTPUT_NAME "${output_name}")
  endif()
endfunction()

cmake_language(DEFER CALL therock_patch_comgr_win_output_name)
