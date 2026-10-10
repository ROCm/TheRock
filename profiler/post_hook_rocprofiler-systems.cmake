# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Dyninst and rocprofiler-systems internal libraries live in lib/rocprofiler-systems.
# Set the RPATH origin per-target so that (for example) lib/rocm_sysdeps/lib resolves as
# $ORIGIN/../rocm_sysdeps/lib rather than $ORIGIN/rocm_sysdeps/lib.
set(_rocprofsys_lib_targets
  common
  dynElf
  dynDwarf
  dyninstAPI
  dynC_API
  instructionAPI
  parseAPI
  patchAPI
  pcontrol
  stackwalk
  symtabAPI
  gotcha
)

foreach(_target ${_rocprofsys_lib_targets})
  if(TARGET "${_target}")
    set_target_properties("${_target}" PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN lib/rocprofiler-systems
    )
  endif()
endforeach()

# libdyninstAPI_RT.so is installed as a symlink pointing to
# ../librocprof-sys-rt.so.X.Y.Z, not as a Dyninst-built library.
# rocprofiler-systems builds its own runtime library (rocprofiler-systems-rt-library)
# installed as librocprof-sys-rt.so in lib/, with a compatibility symlink in
# lib/rocprofiler-systems/libdyninstAPI_RT.so. Since the real file lives in lib/,
# the THEROCK_INSTALL_RPATH_ORIGIN mechanism cannot be used directly (it would
# generate $ORIGIN/.. paths relative to lib/ rather than lib/rocprofiler-systems/).
# Instead, we replace the symlink with a real copy at install time and patch its
# RUNPATH to match the $ORIGIN-relative pattern expected for libraries in
# lib/rocprofiler-systems/.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(DEFINED ENV{PATCHELF})
    set(_therock_rocprofsys_patchelf "$ENV{PATCHELF}")
  else()
    find_program(_therock_rocprofsys_patchelf patchelf)
  endif()
  if(_therock_rocprofsys_patchelf)
    set(_therock_dyninstRT_rpath
        "$ORIGIN/..:$ORIGIN:$ORIGIN/../llvm/lib:$ORIGIN/../rocm_sysdeps/lib")
    install(CODE "
      set(_rt_link \"\${CMAKE_INSTALL_PREFIX}/lib/rocprofiler-systems/libdyninstAPI_RT.so\")
      if(IS_SYMLINK \"\${_rt_link}\")
        get_filename_component(_rt_real \"\${_rt_link}\" REALPATH)
        file(REMOVE \"\${_rt_link}\")
        execute_process(
          COMMAND \"\${CMAKE_COMMAND}\" -E copy \"\${_rt_real}\" \"\${_rt_link}\"
          RESULT_VARIABLE _copy_result
        )
        if(_copy_result EQUAL 0)
          execute_process(
            COMMAND \"${_therock_rocprofsys_patchelf}\" --set-rpath
              \"${_therock_dyninstRT_rpath}\" \"\${_rt_link}\"
            RESULT_VARIABLE _patch_result
          )
          if(NOT _patch_result EQUAL 0)
            message(WARNING \"Failed to patch RUNPATH of \${_rt_link}\")
          endif()
        else()
          message(WARNING \"Failed to copy \${_rt_real} to \${_rt_link}\")
        endif()
      endif()
    ")
  else()
    message(WARNING "patchelf not found; RUNPATH of libdyninstAPI_RT.so will not be "
      "patched. Set the PATCHELF environment variable to the patchelf binary path.")
  endif()
endif()
