# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Windows Control Flow Guard (CFG).
#
# CFG is a Windows exploit mitigation that checks the target of every indirect
# call against a table of legal targets embedded in the image:
# https://learn.microsoft.com/en-us/windows/win32/secbp/control-flow-guard
#
# It needs both a compile flag (the compiler records the legal targets and
# emits the check) and a link flag (the linker writes the guard tables into the
# PE load config), so it has to be applied to every sub-project rather than to
# a single final link. TheRock compiles sub-projects with two different
# compilers, which spell the compile flag differently:
#   * the host MSVC toolchain    -> /guard:cf
#   * the in-tree amd-llvm clang -> -Xclang -cfguard
#
# Support is probed rather than assumed. A toolchain that does not implement
# the flags gets a warning and a build without the mitigation: CFG is defense
# in depth, and losing it is not worth failing an otherwise good build over.
#
# This module is included both by the super-project and by every generated
# sub-project toolchain file, so it must not do anything at include time.
# Toolchain files are re-read by every try_compile, which rules out
# check_<lang>_compiler_flag() there; therock_probe_control_flow_guard() runs
# the compiler directly instead.

# Deliberately no include_guard(): generated toolchain files include this file,
# and CMake re-reads a toolchain file in several different scopes per configure.
# A global guard would leave the variables below set in whichever scope happened
# to include it first. Re-running the set()s and re-defining the functions is
# cheap.

# Compile flag spellings, by compiler. Each is a list, since clang needs two
# arguments.
set(THEROCK_CONTROL_FLOW_GUARD_MSVC_COMPILE_FLAGS "/guard:cf")
# clang exposes CFG as clang-cl's /guard:cf and as the cc1 flag -cfguard.
# TheRock drives the amd-llvm clang through the GNU-style driver, which accepts
# neither directly, so the cc1 flag is passed explicitly. Note that -mguard=cf
# is *not* a substitute: it is only implemented for the ARM and PowerPC
# targets and clang rejects it outright for x86_64-pc-windows-msvc.
set(THEROCK_CONTROL_FLOW_GUARD_CLANG_COMPILE_FLAGS -Xclang -cfguard)
# The link flag goes through CMake's LINKER: prefix, which handles the compiler
# driver difference. /machine and /DYNAMICBASE are omitted: CMake emits the
# former and the linker defaults to the latter.
set(THEROCK_CONTROL_FLOW_GUARD_LINK_FLAGS "/guard:cf")

# therock_control_flow_guard_compile_flags(out_var compiler_id)
# Sets `out_var` to the CFG compile flags for `compiler_id` as a list, or to
# the empty string if we do not know how to spell them for that compiler.
function(therock_control_flow_guard_compile_flags out_var compiler_id)
  if(compiler_id STREQUAL "MSVC")
    set("${out_var}" "${THEROCK_CONTROL_FLOW_GUARD_MSVC_COMPILE_FLAGS}" PARENT_SCOPE)
  elseif(compiler_id MATCHES "Clang")
    set("${out_var}" "${THEROCK_CONTROL_FLOW_GUARD_CLANG_COMPILE_FLAGS}" PARENT_SCOPE)
  else()
    set("${out_var}" "" PARENT_SCOPE)
  endif()
endfunction()

# therock_probe_control_flow_guard(out_var compiler compile_flags)
# Syntax-checks a trivial translation unit with `compile_flags` (a list) and
# sets `out_var` to whether `compiler` accepted them. The result is memoized in
# the cache, keyed on the compiler and the flags.
#
# This exists instead of check_cxx_compiler_flag() because it is called from
# generated toolchain files, which CMake re-includes inside try_compile: the
# check_* modules would recurse.
function(therock_probe_control_flow_guard out_var compiler compile_flags)
  string(JOIN " " _flags_pretty ${compile_flags})
  string(MAKE_C_IDENTIFIER "THEROCK_FLAG_PROBE_${compiler}_${_flags_pretty}" _cache_var)
  if(DEFINED CACHE{${_cache_var}})
    set("${out_var}" "${${_cache_var}}" PARENT_SCOPE)
    return()
  endif()

  if(_flags_pretty MATCHES "^/")
    # MSVC: /Zs is a syntax-only check and /TP forces C++ for a .cpp-agnostic
    # invocation. cl.exe reports an unknown option as warning D9002 and still
    # exits 0, so the output has to be inspected as well.
    set(_probe_args /nologo /Zs /TP)
  else()
    set(_probe_args -fsyntax-only -x c++)
  endif()

  # Resolved from the defining file rather than a module-scope variable, since
  # this may be called from a toolchain file scope that never saw the include.
  set(_probe_source
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/therock_compiler_flag_probe.cpp")

  execute_process(
    COMMAND "${compiler}" ${_probe_args} ${compile_flags} "${_probe_source}"
    RESULT_VARIABLE _probe_rc
    OUTPUT_VARIABLE _probe_stdout
    ERROR_VARIABLE _probe_stderr
  )

  set(_supported TRUE)
  if(NOT _probe_rc EQUAL 0)
    set(_supported FALSE)
  elseif("${_probe_stdout}${_probe_stderr}" MATCHES
      "D9002|unknown argument|unsupported option|unrecognized|is not supported")
    set(_supported FALSE)
  endif()

  set("${_cache_var}" "${_supported}" CACHE INTERNAL
    "Whether ${compiler} accepts ${_flags_pretty}")
  set("${out_var}" "${_supported}" PARENT_SCOPE)
endfunction()

# therock_configure_control_flow_guard()
# Super-project entry point. Declares the THEROCK_ENABLE_CONTROL_FLOW_GUARD
# user option and sets THEROCK_CONTROL_FLOW_GUARD_ACTIVE in the caller's scope
# to the effective state after probing the host toolchain. Must be called after
# the C/CXX compilers have been detected.
#
# The rest of the build keys off THEROCK_CONTROL_FLOW_GUARD_ACTIVE, so that a
# failed probe does not permanently clear the user's option in the cache.
function(therock_configure_control_flow_guard)
  if(WIN32)
    set(_default ON)
  else()
    set(_default OFF)
  endif()
  option(THEROCK_ENABLE_CONTROL_FLOW_GUARD
    "Build Windows binaries with the Control Flow Guard exploit mitigation"
    "${_default}")
  set(THEROCK_CONTROL_FLOW_GUARD_ACTIVE OFF PARENT_SCOPE)

  if(NOT THEROCK_ENABLE_CONTROL_FLOW_GUARD)
    return()
  endif()
  if(NOT WIN32)
    message(WARNING
      "THEROCK_ENABLE_CONTROL_FLOW_GUARD is a Windows-only mitigation and is "
      "ignored on this platform (CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}).")
    return()
  endif()

  therock_control_flow_guard_compile_flags(_compile_flags "${CMAKE_CXX_COMPILER_ID}")
  string(JOIN " " _compile_flags_pretty ${_compile_flags})
  set(_unavailable_reason)
  if(NOT _compile_flags)
    set(_unavailable_reason
      "no Control Flow Guard compile flag is known for compiler id '${CMAKE_CXX_COMPILER_ID}'")
  else()
    include(CheckCXXCompilerFlag)
    include(CheckLinkerFlag)
    check_cxx_compiler_flag("${_compile_flags_pretty}"
      THEROCK_HAVE_CONTROL_FLOW_GUARD_COMPILE_FLAG)
    check_linker_flag(CXX "LINKER:${THEROCK_CONTROL_FLOW_GUARD_LINK_FLAGS}"
      THEROCK_HAVE_CONTROL_FLOW_GUARD_LINK_FLAG)
    if(NOT THEROCK_HAVE_CONTROL_FLOW_GUARD_COMPILE_FLAG)
      set(_unavailable_reason "the compiler rejected ${_compile_flags_pretty}")
    elseif(NOT THEROCK_HAVE_CONTROL_FLOW_GUARD_LINK_FLAG)
      set(_unavailable_reason
        "the linker rejected ${THEROCK_CONTROL_FLOW_GUARD_LINK_FLAGS}")
    endif()
  endif()

  if(_unavailable_reason)
    message(WARNING
      "Control Flow Guard was requested but is not available: "
      "${_unavailable_reason}.\n"
      "  CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}\n"
      "  CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}\n"
      "  CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}\n"
      "Continuing without the mitigation. Set "
      "-DTHEROCK_ENABLE_CONTROL_FLOW_GUARD=OFF to silence this warning.")
    return()
  endif()

  message(STATUS "Control Flow Guard: enabled (${_compile_flags_pretty})")
  set(THEROCK_CONTROL_FLOW_GUARD_ACTIVE ON PARENT_SCOPE)
endfunction()

# therock_toolchain_enable_control_flow_guard(compile_flags subproject)
# Called from a generated sub-project toolchain file, after that file has
# selected the compiler. Probes the compiler and, if it accepts the flags,
# appends them to CMAKE_{C,CXX}_FLAGS_INIT. Either way it sets the
# THEROCK_CONTROL_FLOW_GUARD_SUPPORTED cache entry, which the generated
# project_init.cmake reads to decide whether to add the matching link flags.
#
# This is a macro so that the CMAKE_*_FLAGS_INIT appends land directly in the
# toolchain file's scope, where CMake reads them.
macro(therock_toolchain_enable_control_flow_guard compile_flags subproject)
  therock_probe_control_flow_guard(_therock_cfg_supported
    "${CMAKE_CXX_COMPILER}" "${compile_flags}")
  string(JOIN " " _therock_cfg_flags ${compile_flags})
  if(_therock_cfg_supported)
    string(APPEND CMAKE_C_FLAGS_INIT " ${_therock_cfg_flags}")
    string(APPEND CMAKE_CXX_FLAGS_INIT " ${_therock_cfg_flags}")
  else()
    message(WARNING
      "Control Flow Guard is not supported by the compiler used for "
      "sub-project '${subproject}' and will not be enabled for it:\n"
      "  Compiler: ${CMAKE_CXX_COMPILER}\n"
      "  Rejected flags: ${_therock_cfg_flags}\n"
      "Continuing without the mitigation. Set "
      "-DTHEROCK_ENABLE_CONTROL_FLOW_GUARD=OFF to silence this warning.")
  endif()
  set(THEROCK_CONTROL_FLOW_GUARD_SUPPORTED "${_therock_cfg_supported}"
    CACHE INTERNAL "Whether Control Flow Guard is active for this sub-project")
  unset(_therock_cfg_supported)
  unset(_therock_cfg_flags)
endmacro()
