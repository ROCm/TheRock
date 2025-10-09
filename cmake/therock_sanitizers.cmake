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
  set(_stanza)
  if(_sanitizer STREQUAL "ASAN")
    string(APPEND _stanza "set(THEROCK_SANITIZER \"ASAN\")\n")
    # TODO: Support ASAN_STATIC to use static ASAN linkage. Shared is almost always the right thing,
    # so make "ASAN" imply shared linkage.
    string(APPEND _stanza "string(APPEND CMAKE_CXX_FLAGS \" -fsanitize=address -fno-omit-frame-pointer -g\")\n")
    # Sharp edge: The -shared-libsan flag is compiler frontend specific:
    #   gcc (and gfortran): defaults to shared sanitizer linkage
    #   clang: defaults to static linkage and requires -shared-libsan to link shared
    # This becomes an issue in projects that build with clang and gfortran, so we have to
    # use a generator expression to target the -shared-libsan flag only to clang.
    string(APPEND _stanza "add_link_options(-fsanitize=address\n")
    string(APPEND _stanza "  $<$<AND:$<OR:$<LINK_LANGUAGE:CXX>,$<LINK_LANGUAGE:C>>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-shared-libsan>)\n")
    # Filter GPU_TARGETS to enable xnack+ mode only for gfx targets that support it.
    string(APPEND _stanza "list(TRANSFORM GPU_TARGETS REPLACE \"^(gfx942|gfx950)$\" \"\\\\1:xnack+\")\n")
    string(APPEND _stanza "set(AMDGPU_TARGETS \"\${GPU_TARGETS}\")\n")
    string(APPEND _stanza "message(STATUS \"Override ASAN GPU_TARGETS = \${GPU_TARGETS}\")\n")
    # Action at a distance: Signal that the sub-project should extend its build and install
    # RPATHs to include the clang resource dir.
    string(APPEND _stanza "set(THEROCK_INCLUDE_CLANG_RESOURCE_DIR_RPATH ON)")
  else()
    message(FATAL_ERROR "Cannot configure sanitizer '${_sanitizer} for ${subprojet_name}: unknown sanitizer")
  endif()

  set("${out_sanitizer_stanza}" "${_stanza}" PARENT_SCOPE)
  set("${out_sanitizer_selected}" "${_sanitizer}" PARENT_SCOPE)
endfunction()
