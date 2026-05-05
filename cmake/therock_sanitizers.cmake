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
  set(_stanza)
  if(_sanitizer STREQUAL "ASAN" OR _sanitizer STREQUAL "HOST_ASAN")
    string(APPEND _stanza "set(THEROCK_SANITIZER \"${_sanitizer}\")\n")
    # TODO: Support ASAN_STATIC to use static ASAN linkage. Shared is almost always the right thing,
    # so make "ASAN" imply shared linkage.
    if(_sanitizer STREQUAL "HOST_ASAN")
      # Confine -fsanitize=address to the host compilation pass only.
      # -Xarch_host passes the next flag to the host pass exclusively; the device
      # pass never sees it and never begins ASAN metadata accounting, which would
      # otherwise corrupt HIP fat binaries on targets without xnack+ (e.g. gfx942).
      #
      # CRITICAL: Use add_compile_options, NOT CMAKE_CXX_FLAGS_INIT.
      # CMAKE_CXX_FLAGS is passed by CMake to the linker as <FLAGS> in shared-library
      # link rules. When clang++ links a HIP target (via --hip-link), it receives
      # CMAKE_CXX_FLAGS and processes -fsanitize=address globally — even if preceded
      # by -Xarch_host — because -Xarch_host is a compile-time flag with no defined
      # meaning at link time. This causes the same fat binary corruption as bare
      # -fsanitize=address in the link command.
      # add_compile_options populates COMPILE_OPTIONS (compile-only; never passed to
      # the linker), so -Xarch_host -fsanitize=address is cleanly confined to
      # compilation steps. The link side is handled separately below by
      # directly linking the ASAN runtime library (bypassing -fsanitize=address).
      string(APPEND _stanza "add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Xarch_host>\n")
      string(APPEND _stanza "  $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>\n")
      string(APPEND _stanza "  $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>\n")
      string(APPEND _stanza "  $<$<COMPILE_LANGUAGE:CXX>:-g>)\n")
    else()
      string(APPEND _stanza "string(APPEND CMAKE_CXX_FLAGS_INIT \" -fsanitize=address -fno-omit-frame-pointer -g\")\n")
    endif()
    string(APPEND _stanza "string(APPEND CMAKE_C_FLAGS_INIT \" -fsanitize=address -fno-omit-frame-pointer -g\")\n")
    # Sharp edge: The -shared-libsan flag is compiler frontend specific:
    #   gcc (and gfortran): defaults to shared sanitizer linkage
    #   clang: defaults to static linkage and requires -shared-libsan to link shared
    # This becomes an issue in projects that build with clang and gfortran, so we have to
    # use a generator expression to target the -shared-libsan flag only to clang.
    # Only enable ASAN for C/C++ for now. Include fortran once the toolchain
    # is available and can be used for portable builds.
    # https://github.com/ROCm/TheRock/issues/1782
    #
    # HOST_ASAN link-flag sharp edge: clang++ in HIP link mode (triggered by
    # hip::device injecting --hip-link via INTERFACE_LINK_LIBRARIES) processes
    # ALL link flags — including -fsanitize=address — globally across host and
    # device sections. -Xarch_host has NO defined meaning at link time, so it
    # cannot confine -fsanitize=address to the host link pass.
    #
    # Any form of -fsanitize=address reaching clang++ during --hip-link causes
    # ASAN metadata accounting in the device fat binary sections, corrupting
    # .hipFatBinSegment on targets without xnack+ (e.g. gfx942).
    #
    # Solution: bypass -fsanitize=address entirely at link time. Instead, find
    # the ASAN shared runtime library and link it directly. This gives us ASAN
    # host-side linking without triggering clang's device-side ASAN processing.
    # For shared ASAN, the runtime initializes via __attribute__((constructor))
    # when the library is loaded, so explicit -fsanitize=address is not needed.
    #
    # C targets still get -fsanitize=address via CMAKE_C_FLAGS (from
    # CMAKE_C_FLAGS_INIT above) which bleeds to the C linker. This is fine
    # because C targets don't use --hip-link mode.
    if(_sanitizer STREQUAL "HOST_ASAN")
      string(APPEND _stanza "execute_process(\n")
      string(APPEND _stanza "  COMMAND \"\${CMAKE_CXX_COMPILER}\" --print-file-name=libclang_rt.asan.so\n")
      string(APPEND _stanza "  OUTPUT_VARIABLE _therock_asan_lib\n")
      string(APPEND _stanza "  OUTPUT_STRIP_TRAILING_WHITESPACE)\n")
      string(APPEND _stanza "if(NOT EXISTS \"\${_therock_asan_lib}\")\n")
      string(APPEND _stanza "  execute_process(\n")
      string(APPEND _stanza "    COMMAND \"\${CMAKE_CXX_COMPILER}\" --print-file-name=libclang_rt.asan-\${CMAKE_HOST_SYSTEM_PROCESSOR}.so\n")
      string(APPEND _stanza "    OUTPUT_VARIABLE _therock_asan_lib\n")
      string(APPEND _stanza "    OUTPUT_STRIP_TRAILING_WHITESPACE)\n")
      string(APPEND _stanza "endif()\n")
      string(APPEND _stanza "if(EXISTS \"\${_therock_asan_lib}\")\n")
      string(APPEND _stanza "  message(STATUS \"HOST_ASAN_DIAG: Direct-linking ASAN runtime: \${_therock_asan_lib}\")\n")
      string(APPEND _stanza "  link_libraries(\"\${_therock_asan_lib}\")\n")
      string(APPEND _stanza "else()\n")
      string(APPEND _stanza "  message(FATAL_ERROR \"HOST_ASAN: Cannot find ASAN runtime library via --print-file-name\")\n")
      string(APPEND _stanza "endif()\n")
    else()
      string(APPEND _stanza "add_link_options($<$<LINK_LANGUAGE:C,CXX>:-fsanitize=address>\n")
      string(APPEND _stanza "  $<$<AND:$<LINK_LANGUAGE:C,CXX>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-shared-libsan>)\n")
    endif()
    # Device-side ASAN: Only for full ASAN mode, not HOST_ASAN.
    # Filter GPU_TARGETS to enable xnack+ mode only for gfx targets that support it.
    if(_sanitizer STREQUAL "ASAN")
      string(APPEND _stanza "list(TRANSFORM GPU_TARGETS REPLACE \"^(gfx942|gfx950)$\" \"\\\\1:xnack+\")\n")
      string(APPEND _stanza "set(AMDGPU_TARGETS \"\${GPU_TARGETS}\")\n")
      string(APPEND _stanza "message(STATUS \"Override ASAN GPU_TARGETS = \${GPU_TARGETS}\")\n")
    else()
      string(APPEND _stanza "message(STATUS \"HOST_ASAN enabled - GPU_TARGETS unchanged\")\n")
    endif()
    # Verbose build output for HOST_ASAN on rand subprojects so the full
    # compile and link commands appear in the build log for diagnostics.
    if(_sanitizer STREQUAL "HOST_ASAN" AND subproject_name MATCHES "rocRAND|hipRAND")
      string(APPEND _stanza "set(CMAKE_VERBOSE_MAKEFILE ON)\n")
      string(APPEND _stanza "message(STATUS \"HOST_ASAN_DIAG: Verbose build enabled for ${subproject_name}\")\n")
    endif()
    # Action at a distance: Signal that the sub-project should extend its build and install
    # RPATHs to include the clang resource dir.
    string(APPEND _stanza "set(THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH ON)")
  elseif(_sanitizer STREQUAL "TSAN")
    string(APPEND _stanza "set(THEROCK_SANITIZER \"TSAN\")\n")
    # TODO: Support TSAN_STATIC to use static TSAN linkage. Shared is almost always the right thing,
    # so make "TSAN" imply shared linkage.
    string(APPEND _stanza "string(APPEND CMAKE_CXX_FLAGS_INIT \" -fsanitize=thread -fno-omit-frame-pointer -g\")\n")
    string(APPEND _stanza "string(APPEND CMAKE_C_FLAGS_INIT \" -fsanitize=thread -fno-omit-frame-pointer -g\")\n")
    # Sharp edge: The -shared-libsan flag is compiler frontend specific:
    #   gcc (and gfortran): defaults to shared sanitizer linkage
    #   clang: defaults to static linkage and requires -shared-libsan to link shared
    # This becomes an issue in projects that build with clang and gfortran, so we have to
    # use a generator expression to target the -shared-libsan flag only to clang.
    # Only enable TSAN for C/C++ for now. Include fortran once the toolchain
    # is available and can be used for portable builds.
    # https://github.com/ROCm/TheRock/issues/1782
    string(APPEND _stanza "add_link_options($<$<LINK_LANGUAGE:C,CXX>:-fsanitize=thread>\n")
    string(APPEND _stanza "  $<$<AND:$<LINK_LANGUAGE:C,CXX>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-shared-libsan>)\n")
    # Filter GPU_TARGETS to enable xnack+ mode only for gfx targets that support it.
    string(APPEND _stanza "list(TRANSFORM GPU_TARGETS REPLACE \"^(gfx942|gfx950)$\" \"\\\\1:xnack+\")\n")
    string(APPEND _stanza "set(AMDGPU_TARGETS \"\${GPU_TARGETS}\")\n")
    string(APPEND _stanza "message(STATUS \"Override TSAN GPU_TARGETS = \${GPU_TARGETS}\")\n")
    # Action at a distance: Signal that the sub-project should extend its build and install
    # RPATHs to include the clang resource dir.
    string(APPEND _stanza "set(THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH ON)")
  else()
    message(FATAL_ERROR "Cannot configure sanitizer '${_sanitizer}' for ${subproject_name}: unknown sanitizer")
  endif()

  set("${out_sanitizer_stanza}" "${_stanza}" PARENT_SCOPE)
  set("${out_sanitizer_selected}" "${_sanitizer}" PARENT_SCOPE)
endfunction()
