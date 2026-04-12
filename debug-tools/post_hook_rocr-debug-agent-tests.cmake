# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The rocm-debug-agent-test executable is transitioning from bin/ to
# tests/rocm-debug-agent-tests/bin/. For compatibility with both locations
# during the transition, we set RUNPATH entries for both old (bin/) and
# new (tests/rocm-debug-agent-tests/) directory structures.
#
# This post hook will be removed when the transition is done.
if(TARGET rocm-debug-agent-test)
  # Use cmake_language(DEFER) to set RPATH after therock_global_post_subproject.cmake runs.
  # This ensures our custom RPATH isn't overwritten by the automated processing.
  cmake_language(DEFER CALL set_target_properties rocm-debug-agent-test PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib/rocm_sysdeps/lib:$ORIGIN/../lib/llvm/lib:$ORIGIN/../lib:$ORIGIN/../../lib/rocm_sysdeps/lib:$ORIGIN/../../lib/llvm/lib:$ORIGIN/../../lib"
  )
  cmake_language(DEFER CALL message STATUS
    "Set custom transitional RPATH on rocm-debug-agent-test: ${INSTALL_RPATH}"
  )
endif()
