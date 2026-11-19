# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Provider targets are shared by core, profiling and test subprojects.
set(THEROCK_HSA_RUNTIME_TARGET ROCR-Runtime)
set(THEROCK_HSAKMT_TARGET ROCR-Runtime)
if(THEROCK_FLAG_USE_ROCDDI_FOR_HSA_RUNTIME)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Rocddi HSA selection currently requires Linux")
  endif()
  if(NOT THEROCK_ENABLE_ROCDDI_RUNTIMES OR NOT THEROCK_ENABLE_CORE_RUNTIME)
    message(FATAL_ERROR
      "Rocddi HSA selection requires THEROCK_ENABLE_ROCDDI_RUNTIMES=ON and THEROCK_ENABLE_CORE_RUNTIME=ON")
  endif()
  set(THEROCK_HSA_RUNTIME_TARGET rocddi-hsa)
  set(THEROCK_HSAKMT_TARGET hsakmt)
endif()

# Both provider configurations produce the same core-runtime artifact identity.
# Derive common tool/HRX ownership from its one maintained source descriptor.
function(therock_configure_hsa_artifact source output)
  file(READ "${source}" _descriptor)
  string(REPLACE "core/ROCR-Runtime/stage" "core/rocddi-hsa/stage" _descriptor "${_descriptor}")
  if(TARGET hsakmt)
    string(APPEND _descriptor "
[components.lib.\"core/hsakmt/stage\"]
[components.dev.\"core/hsakmt/stage\"]
[components.doc.\"core/hsakmt/stage\"]
")
  endif()
  configure_file("${source}" "${output}.input" COPYONLY)
  file(CONFIGURE OUTPUT "${output}" CONTENT "${_descriptor}" @ONLY)
endfunction()
