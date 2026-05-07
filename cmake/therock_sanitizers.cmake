# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

function(therock_sanitizer_configure
    out_sanitizer_stanza
    out_sanitizer_selected
    cxx_compiler
    compiler_toolchain
    subproject_name)
  # Use global sanitizer setting unless if defined for a sub-project.
  set(_sanitizer "${THEROCK_SANITIZER}")
  if(DEFINED "${subproject_name}_SANITIZER")
    set(_sanitizer "${${subproject_name}_SANITIZER}")
  endif()

  # Default disabled output.
  set("${out_sanitizer_stanza}" "" PARENT_SCOPE)
  set("${out_sanitizer_selected}" "" PARENT_SCOPE)

  # Disabled.
  if(NOT _sanitizer)
    return()
  endif()

  # Enabled.
  if(NOT compiler_toolchain)
    message(WARNING "Sub-project ${subproject_name} built with the system toolchain does not support sanitizer ${_sanitizer}")
    return()
  endif()

  # Our own toolchains get ASAN enabled consistently.
  # ASAN: Full host+device address sanitizer (xnack+ GPU targets for gfx942, gfx950)
  # HOST_ASAN: Host-only address sanitizer (no device-side instrumentation)
  # TSAN: Thread sanitizer.
  set(_stanza)
  if(_sanitizer STREQUAL "ASAN" OR _sanitizer STREQUAL "HOST_ASAN" OR _sanitizer STREQUAL "TSAN")
    string(APPEND _stanza "set(THEROCK_SANITIZER \"${_sanitizer}\")\n")

    # Set the compiler sanitizer string for the command line.
    set(_sanitizer_string "address")
    if(_sanitizer STREQUAL "TSAN")
      set(_sanitizer_string "thread")
    endif()

    # TODO: Support ASAN_STATIC/TSAN_STATIC to use static sanitizer linkage. Shared is almost always the right thing,
    # so make the sanitizer imply shared linkage.
    string(APPEND _stanza "string(APPEND CMAKE_CXX_FLAGS_INIT \" -fsanitize=${_sanitizer_string} -fno-omit-frame-pointer -g\")\n")
    string(APPEND _stanza "string(APPEND CMAKE_C_FLAGS_INIT \" -fsanitize=${_sanitizer_string} -fno-omit-frame-pointer -g\")\n")

    # Sharp edge: The -shared-libsan flag is compiler frontend specific:
    #   gcc (and gfortran): defaults to shared sanitizer linkage
    #   clang: defaults to static linkage and requires -shared-libsan to link shared
    # This becomes an issue in projects that build with clang and gfortran, so we have to
    # use a generator expression to target the -shared-libsan flag only to clang.
    # Only enable sanitizers for C/C++ for now. Include fortran once the toolchain
    # is available and can be used for portable builds.
    # https://github.com/ROCm/TheRock/issues/1782
    string(APPEND _stanza "add_link_options($<$<LINK_LANGUAGE:C,CXX>:-fsanitize=${_sanitizer_string}>\n")
    string(APPEND _stanza "  $<$<AND:$<LINK_LANGUAGE:C,CXX>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-shared-libsan>)\n")

    # For autotools-based builds (like rocgdb) that use CMAKE_*_LINKER_FLAGS directly,
    # always include -fsanitize=<sanitizer> for linking. Also add -shared-libsan since
    # we know we are dealing with a non-system compiler.
    string(APPEND _stanza "string(APPEND CMAKE_EXE_LINKER_FLAGS \" -fsanitize=${_sanitizer_string} -shared-libsan\")\n")
    string(APPEND _stanza "string(APPEND CMAKE_SHARED_LINKER_FLAGS \" -fsanitize=${_sanitizer_string} -shared-libsan\")\n")

    # Device-side ASAN: Only for full ASAN mode, not HOST_ASAN.
    # Filter GPU_TARGETS to enable xnack+ mode only for gfx targets that support it.
    if(_sanitizer STREQUAL "ASAN" OR _sanitizer STREQUAL "TSAN")
      # ASAN and TSAN.
      string(APPEND _stanza "list(TRANSFORM GPU_TARGETS REPLACE \"^(gfx942|gfx950)$\" \"\\\\1:xnack+\")\n")
      string(APPEND _stanza "set(AMDGPU_TARGETS \"\${GPU_TARGETS}\")\n")
      string(APPEND _stanza "message(STATUS \"Override \${_sanitizer} GPU_TARGETS = \${GPU_TARGETS}\")\n")
    else()
      # HOST_ASAN.
      string(APPEND _stanza "message(STATUS \"HOST_ASAN enabled - GPU_TARGETS unchanged\")\n")
    endif()

    # Action at a distance: Signal that the sub-project should extend its build and install
    # RPATHs to include the clang resource dir.
    string(APPEND _stanza "set(THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH ON)")
  else()
    message(FATAL_ERROR "Cannot configure sanitizer '${_sanitizer}' for ${subproject_name}: unknown sanitizer")
  endif()

  set("${out_sanitizer_stanza}" "${_stanza}" PARENT_SCOPE)
  set("${out_sanitizer_selected}" "${_sanitizer}" PARENT_SCOPE)
endfunction()
