# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Host compiler and linker hardening policy for code built by TheRock.
#
# Ordinary CMake subprojects load this module from their generated _init.cmake;
# the global post hook applies the policy after project and user hooks have
# defined their targets. Independent inner builds either use the command flag
# helper below or opt into the deferred CMAKE_PROJECT_INCLUDE path at the end.

cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

set(_THEROCK_HARDENING_POLICIES OFF BASELINE STRICT)
# These producer subprojects are deferred to a separately reviewed compiler
# hardening effort. Consumers built with amd-llvm remain in scope.
set(_THEROCK_HARDENING_COMPILER_SUBPROJECTS
  amd-llvm
  amd-comgr
  hipcc
  hipify
)
set(_THEROCK_HARDENING_COMPILE_OPTIONS
  -fstack-protector-strong
  -fstack-clash-protection
  -fcf-protection=full
  -Wformat
  -Wformat-security
)
# Fortification is added separately because it depends on configuration and
# sanitizer state. PIE is likewise limited to executable targets.
# The driver reset suppresses compiler defaults; the forwarded reset also
# clears inherited -Wp definitions before the policy-owned definition.
set(_THEROCK_HARDENING_FORTIFY_RESET_OPTIONS
  -U_FORTIFY_SOURCE -Wp,-U_FORTIFY_SOURCE
)
set(_THEROCK_HARDENING_FORTIFY_OPTIONS
  ${_THEROCK_HARDENING_FORTIFY_RESET_OPTIONS} -Wp,-D_FORTIFY_SOURCE=3
)
set(_THEROCK_HARDENING_LINK_OPTIONS
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
)

# Validates the selected policy and its current platform constraints.
function(_therock_validate_hardening_policy)
  string(TOUPPER "${THEROCK_HARDENING_POLICY}" _policy)
  if(NOT _policy IN_LIST _THEROCK_HARDENING_POLICIES)
    message(FATAL_ERROR
      "Invalid THEROCK_HARDENING_POLICY='${THEROCK_HARDENING_POLICY}'. "
      "Supported values are OFF, BASELINE, and STRICT."
    )
  endif()
  if(_policy STREQUAL "STRICT")
    message(FATAL_ERROR
      "THEROCK_HARDENING_POLICY=STRICT is not implemented. Its protections "
      "and validation criteria must be defined before it can be selected."
    )
  endif()
  if(_policy STREQUAL "BASELINE")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _processor)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
        NOT _processor MATCHES "^(x86_64|amd64)$")
      message(FATAL_ERROR
        "THEROCK_HARDENING_POLICY=BASELINE currently supports only Linux "
        "x86-64 hosts (got ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR})."
      )
    endif()
  endif()
endfunction()

# Defines, normalizes, and validates the user-facing policy cache variable.
function(therock_initialize_hardening_policy)
  set(THEROCK_HARDENING_POLICY "OFF" CACHE STRING
    "Host compiler and linker hardening policy (OFF, BASELINE, or STRICT)"
  )
  set_property(CACHE THEROCK_HARDENING_POLICY PROPERTY STRINGS
    ${_THEROCK_HARDENING_POLICIES}
  )
  string(TOUPPER "${THEROCK_HARDENING_POLICY}" _policy)
  set(THEROCK_HARDENING_POLICY "${_policy}" CACHE STRING
    "Host compiler and linker hardening policy (OFF, BASELINE, or STRICT)"
    FORCE
  )
  _therock_validate_hardening_policy()
endfunction()

# Reports whether a subproject belongs to the deferred compiler group.
function(therock_is_hardening_excluded_subproject out_var target_name)
  if(target_name IN_LIST _THEROCK_HARDENING_COMPILER_SUBPROJECTS)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Rejects compilers outside the current policy's supported language matrix.
function(_therock_hardening_validate_compiler language compiler_id)
  if(language STREQUAL "HIP")
    set(_supported_id_pattern "^Clang$")
  else()
    set(_supported_id_pattern "^(GNU|Clang)$")
  endif()
  if(NOT compiler_id MATCHES "${_supported_id_pattern}")
    message(FATAL_ERROR
      "THEROCK_HARDENING_POLICY=BASELINE does not support the "
      "${language} compiler '${compiler_id}'."
    )
  endif()
endfunction()

# Confirms that the compiler and C library provide effective fortify level 3.
function(_therock_hardening_validate_fortify language compiler_id binary_dir)
  foreach(_variable
      CMAKE_${language}_COMPILER
      CMAKE_${language}_COMPILER_TARGET
      CMAKE_${language}_FLAGS
      CMAKE_SYSROOT
      CMAKE_HIP_ARCHITECTURES)
    get_directory_property(_${_variable}
      DIRECTORY "${binary_dir}"
      DEFINITION "${_variable}"
    )
  endforeach()

  set(_probe_identity
    "${language};${compiler_id};${_CMAKE_${language}_COMPILER};"
    "${_CMAKE_${language}_COMPILER_TARGET};${_CMAKE_${language}_FLAGS};"
    "${_CMAKE_SYSROOT};${_CMAKE_HIP_ARCHITECTURES}"
  )
  string(SHA256 _probe_key "${_probe_identity}")
  get_property(_already_validated GLOBAL PROPERTY
    "THEROCK_HARDENING_FORTIFY_VALIDATED_${_probe_key}")
  if(_already_validated)
    return()
  endif()

  # Keep the fortify reset and definition ordered through the preprocessor.
  if(language STREQUAL "HIP")
    set(_source_extension hip)
    set(_probe_flags
      -O2
      -Xarch_host -U_FORTIFY_SOURCE
      -Xarch_host -Wp,-U_FORTIFY_SOURCE
      -Xarch_host -Wp,-D_FORTIFY_SOURCE=3
    )
  elseif(language STREQUAL "CXX")
    set(_source_extension cpp)
    set(_probe_flags
      -O2 ${_THEROCK_HARDENING_FORTIFY_OPTIONS}
    )
  else()
    set(_source_extension c)
    set(_probe_flags
      -O2 ${_THEROCK_HARDENING_FORTIFY_OPTIONS}
    )
  endif()

  # A standalone project accepts directory-local flags as cache inputs without
  # assigning watched toolchain variables in the caller. The source-file form
  # of try_compile overwrites such inputs with the caller's flags.
  set(_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/therock_fortify_probe/${_probe_key}")
  string(JOIN " " _probe_options ${_probe_flags})
  file(WRITE "${_probe_dir}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.25)\n"
    "project(therock_fortify_probe LANGUAGES ${language})\n"
    "add_definitions(${_probe_options})\n"
    "add_library(therock_fortify_probe STATIC therock_fortify_probe.${_source_extension})\n"
  )
  file(WRITE "${_probe_dir}/therock_fortify_probe.${_source_extension}" [[
#if !defined(__HIP_DEVICE_COMPILE__)
#include <features.h>
#if !defined(_FORTIFY_SOURCE) || _FORTIFY_SOURCE != 3
#error "_FORTIFY_SOURCE is not 3"
#endif
#if !defined(__USE_FORTIFY_LEVEL) || __USE_FORTIFY_LEVEL != 3
#error "effective fortify level is not 3"
#endif
#endif
int therock_fortify_probe;
]]
  )
  try_compile(_fortify_supported
    PROJECT therock_fortify_probe
    SOURCE_DIR "${_probe_dir}"
    BINARY_DIR "${_probe_dir}/build"
    CMAKE_FLAGS
      "-DCMAKE_${language}_COMPILER:FILEPATH=${_CMAKE_${language}_COMPILER}"
      "-DCMAKE_${language}_COMPILER_TARGET:STRING=${_CMAKE_${language}_COMPILER_TARGET}"
      "-DCMAKE_${language}_FLAGS:STRING=${_CMAKE_${language}_FLAGS}"
      "-DCMAKE_SYSROOT:PATH=${_CMAKE_SYSROOT}"
      "-DCMAKE_HIP_ARCHITECTURES:STRING=${_CMAKE_HIP_ARCHITECTURES}"
    OUTPUT_VARIABLE _probe_output
    NO_CACHE
  )
  if(NOT _fortify_supported)
    message(FATAL_ERROR
      "THEROCK_HARDENING_POLICY=BASELINE requires effective "
      "_FORTIFY_SOURCE=3 for the ${language} compiler '${compiler_id}', but "
      "the compiler and C library did not provide it. Probe output:\n"
      "${_probe_output}"
    )
  endif()
  set_property(GLOBAL PROPERTY
    "THEROCK_HARDENING_FORTIFY_VALIDATED_${_probe_key}" TRUE)
endfunction()

# Validates compilers in the target's directory and optional fortify support.
function(_therock_hardening_validate_target_compilers target validate_fortify)
  # Languages may be enabled only in a child directory, so global compiler
  # variables do not reliably describe the target's effective toolchain.
  get_target_property(_binary_dir "${target}" BINARY_DIR)
  # Validate fortify only if a generated configuration can enable it. This
  # restriction is specific to native targets: an optimized independent build
  # may request fortify even when its outer wrapper uses Debug.
  get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_multi_config)
    get_directory_property(_configurations DIRECTORY "${_binary_dir}"
      DEFINITION CMAKE_CONFIGURATION_TYPES)
  else()
    get_directory_property(_configurations DIRECTORY "${_binary_dir}"
      DEFINITION CMAKE_BUILD_TYPE)
  endif()
  string(TOUPPER "${_configurations}" _configurations)
  if(NOT _configurations MATCHES "(^|;)(RELEASE|RELWITHDEBINFO|MINSIZEREL)(;|$)")
    set(validate_fortify FALSE)
  endif()
  foreach(_language C CXX HIP)
    get_directory_property(_compiler_loaded
      DIRECTORY "${_binary_dir}"
      DEFINITION "CMAKE_${_language}_COMPILER_LOADED"
    )
    if(_compiler_loaded)
      get_directory_property(_compiler_id
        DIRECTORY "${_binary_dir}"
        DEFINITION "CMAKE_${_language}_COMPILER_ID"
      )
      _therock_hardening_validate_compiler("${_language}" "${_compiler_id}")
      if(validate_fortify)
        _therock_hardening_validate_fortify(
          "${_language}" "${_compiler_id}" "${_binary_dir}"
        )
      endif()
    endif()
  endforeach()
endfunction()

# Independent Autotools wrappers can use this to classify explicit CFLAGS.
# Unknown or absent optimization is conservatively treated as unoptimized.
function(therock_hardening_flags_are_optimized out_var flags)
  separate_arguments(_options UNIX_COMMAND "${flags}")
  set(_optimized OFF)
  foreach(_option IN LISTS _options)
    if(_option MATCHES "^-O(1|2|3|fast|s|z)$")
      set(_optimized ON)
    elseif(_option MATCHES "^-O(0|g)$")
      set(_optimized OFF)
    endif()
  endforeach()
  set(${out_var} "${_optimized}" PARENT_SCOPE)
endfunction()

# Adds one private host compile option for each supported language and compiler.
function(_therock_hardening_add_compile_option target option)
  # -Xarch_host prevents Clang options from reaching AMDGPU device compilation.
  target_compile_options("${target}" PRIVATE
    "$<$<COMPILE_LANG_AND_ID:C,GNU>:${option}>"
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:${option}>"
    "$<$<COMPILE_LANG_AND_ID:C,Clang>:SHELL:-Xarch_host ${option}>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:SHELL:-Xarch_host ${option}>"
    "$<$<COMPILE_LANG_AND_ID:HIP,Clang>:SHELL:-Xarch_host ${option}>"
  )
endfunction()

# Adds level-3 fortification to optimized configurations of a target.
function(_therock_hardening_add_fortify target)
  set(_optimized_configuration
    "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>"
  )
  # Keep each undefine/define pair in one SHELL group. Separate options can be
  # reordered or de-duplicated by CMake and leave an inherited definition active.
  string(JOIN " " _reset ${_THEROCK_HARDENING_FORTIFY_RESET_OPTIONS})
  string(JOIN " " _fortify ${_THEROCK_HARDENING_FORTIFY_OPTIONS})
  set(_host_reset)
  foreach(_option IN LISTS _THEROCK_HARDENING_FORTIFY_RESET_OPTIONS)
    string(APPEND _host_reset " -Xarch_host ${_option}")
  endforeach()
  set(_host_fortify "${_host_reset} -Xarch_host -Wp,-D_FORTIFY_SOURCE=3")
  target_compile_options("${target}" PRIVATE
    "$<$<AND:${_optimized_configuration},$<COMPILE_LANG_AND_ID:C,GNU>>:SHELL:${_fortify}>"
    "$<$<AND:${_optimized_configuration},$<COMPILE_LANG_AND_ID:CXX,GNU>>:SHELL:${_fortify}>"
    "$<$<AND:${_optimized_configuration},$<COMPILE_LANG_AND_ID:C,Clang>>:SHELL:${_host_fortify}>"
    "$<$<AND:${_optimized_configuration},$<COMPILE_LANG_AND_ID:CXX,Clang>>:SHELL:${_host_fortify}>"
    "$<$<AND:${_optimized_configuration},$<COMPILE_LANG_AND_ID:HIP,Clang>>:SHELL:${_host_fortify}>"
    "$<$<AND:$<NOT:${_optimized_configuration}>,$<COMPILE_LANG_AND_ID:C,GNU>>:SHELL:${_reset}>"
    "$<$<AND:$<NOT:${_optimized_configuration}>,$<COMPILE_LANG_AND_ID:CXX,GNU>>:SHELL:${_reset}>"
    "$<$<AND:$<NOT:${_optimized_configuration}>,$<COMPILE_LANG_AND_ID:C,Clang>>:SHELL:${_host_reset}>"
    "$<$<AND:$<NOT:${_optimized_configuration}>,$<COMPILE_LANG_AND_ID:CXX,Clang>>:SHELL:${_host_reset}>"
    "$<$<AND:$<NOT:${_optimized_configuration}>,$<COMPILE_LANG_AND_ID:HIP,Clang>>:SHELL:${_host_reset}>"
  )
endfunction()

# Removes inherited fortify definitions for sanitizer and target exceptions.
function(_therock_hardening_add_fortify_reset target)
  # Keep the resets atomic so a matching earlier target option cannot make
  # CMake deduplicate away the final reset after a conflicting definition.
  string(JOIN " " _reset ${_THEROCK_HARDENING_FORTIFY_RESET_OPTIONS})
  set(_host_reset)
  foreach(_option IN LISTS _THEROCK_HARDENING_FORTIFY_RESET_OPTIONS)
    string(APPEND _host_reset " -Xarch_host ${_option}")
  endforeach()
  target_compile_options("${target}" PRIVATE
    "$<$<COMPILE_LANG_AND_ID:C,GNU>:SHELL:${_reset}>"
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:SHELL:${_reset}>"
    "$<$<COMPILE_LANG_AND_ID:C,Clang>:SHELL:${_host_reset}>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:SHELL:${_host_reset}>"
    "$<$<COMPILE_LANG_AND_ID:HIP,Clang>:SHELL:${_host_reset}>"
  )
endfunction()

# Adds one private host link option for each supported link language.
function(_therock_hardening_add_link_option target option)
  target_link_options("${target}" PRIVATE
    "$<$<LINK_LANG_AND_ID:C,GNU>:${option}>"
    "$<$<LINK_LANG_AND_ID:CXX,GNU>:${option}>"
    "$<$<LINK_LANG_AND_ID:C,Clang>:SHELL:-Xarch_host ${option}>"
    "$<$<LINK_LANG_AND_ID:CXX,Clang>:SHELL:-Xarch_host ${option}>"
    "$<$<LINK_LANG_AND_ID:HIP,Clang>:SHELL:-Xarch_host ${option}>"
  )
endfunction()

# Reports once when a sanitizer causes BASELINE to omit fortification.
function(_therock_hardening_report_sanitizer)
  if(NOT THEROCK_SANITIZER)
    return()
  endif()
  get_property(_reported GLOBAL PROPERTY
    THEROCK_HARDENING_SANITIZER_REPORTED)
  if(NOT _reported)
    message(STATUS
      "TheRock hardening: omitting _FORTIFY_SOURCE=3 for "
      "THEROCK_SANITIZER=${THEROCK_SANITIZER}; other BASELINE protections "
      "remain enabled"
    )
    set_property(GLOBAL PROPERTY
      THEROCK_HARDENING_SANITIZER_REPORTED TRUE)
  endif()
endfunction()

# Applies the selected policy to eligible targets and explicit exceptions.
function(therock_apply_hardening_to_targets)
  _therock_validate_hardening_policy()
  if(THEROCK_HARDENING_POLICY STREQUAL "OFF" OR
      THEROCK_HARDENING_EXCLUDED)
    return()
  endif()
  _therock_hardening_report_sanitizer()

  foreach(_target IN LISTS ARGV)
    if(NOT TARGET "${_target}")
      message(FATAL_ERROR "Cannot harden unknown target '${_target}'")
    endif()
    get_target_property(_imported "${_target}" IMPORTED)
    get_target_property(_type "${_target}" TYPE)
    # THEROCK_DISABLE_HARDENING is the explicit whole-target exception.
    get_target_property(_disable_hardening "${_target}"
      THEROCK_DISABLE_HARDENING)
    if(_imported OR _disable_hardening OR NOT _type MATCHES
        "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY|STATIC_LIBRARY|OBJECT_LIBRARY)$")
      continue()
    endif()
    # Sanitizers suppress fortification for the subproject. Targets with local
    # -O0, sanitizer, or conflicting fortify flags use the narrower property.
    get_target_property(_disable_fortify "${_target}"
      THEROCK_DISABLE_FORTIFY)
    if(NOT THEROCK_SANITIZER AND NOT _disable_fortify)
      set(_use_fortify TRUE)
    else()
      set(_use_fortify FALSE)
    endif()
    _therock_hardening_validate_target_compilers(
      "${_target}" "${_use_fortify}"
    )

    foreach(_option IN LISTS _THEROCK_HARDENING_COMPILE_OPTIONS)
      _therock_hardening_add_compile_option("${_target}" "${_option}")
    endforeach()

    if(_use_fortify)
      _therock_hardening_add_fortify("${_target}")
    else()
      _therock_hardening_add_fortify_reset("${_target}")
    endif()

    # CMake's property checks transitive PIC compatibility. Explicit host flags
    # preserve policy ordering and keep HIP device compilation unaffected.
    if(_type STREQUAL "EXECUTABLE")
      get_property(_pic_is_set TARGET "${_target}"
        PROPERTY POSITION_INDEPENDENT_CODE SET)
      get_target_property(_pic "${_target}" POSITION_INDEPENDENT_CODE)
      if(_pic_is_set AND NOT _pic)
        message(FATAL_ERROR
          "Cannot harden executable '${_target}': "
          "POSITION_INDEPENDENT_CODE is OFF and conflicts with BASELINE PIE. "
          "Set THEROCK_DISABLE_HARDENING=ON for an explicit target exception."
        )
      endif()
      set_property(TARGET "${_target}"
        PROPERTY POSITION_INDEPENDENT_CODE TRUE)
      _therock_hardening_add_compile_option("${_target}" -fPIE)
    elseif(_type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY)$")
      # Objects that may feed a PIE executable or shared library must be PIC.
      set_property(TARGET "${_target}"
        PROPERTY POSITION_INDEPENDENT_CODE TRUE)
      _therock_hardening_add_compile_option("${_target}" -fPIC)
    endif()
    if(_type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$")
      foreach(_option IN LISTS _THEROCK_HARDENING_LINK_OPTIONS)
        _therock_hardening_add_link_option("${_target}" "${_option}")
      endforeach()
      if(_type STREQUAL "EXECUTABLE")
        _therock_hardening_add_link_option("${_target}" -pie)
      endif()
    endif()
  endforeach()
endfunction()

# Finds and hardens all eligible targets rooted in the current directory.
function(_therock_apply_hardening_to_directory)
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  therock_get_all_targets(_targets "${CMAKE_CURRENT_SOURCE_DIR}")
  therock_apply_hardening_to_targets(${_targets})
endfunction()

# Schedules one directory-wide pass after all targets have been defined.
function(therock_schedule_hardening)
  _therock_validate_hardening_policy()
  get_property(_scheduled GLOBAL PROPERTY
    THEROCK_COMPILER_HARDENING_SCHEDULED)
  if(_scheduled)
    return()
  endif()
  set_property(GLOBAL PROPERTY THEROCK_COMPILER_HARDENING_SCHEDULED TRUE)
  cmake_language(DEFER CALL _therock_apply_hardening_to_directory)
endfunction()

# Returns space-separated flags for build systems invoked from a CMake wrapper.
# The wrapper selects fortification from the inner build's optimization and
# sanitizer settings by passing FORTIFY ON or OFF.
function(therock_get_hardening_command_flags
    out_compile
    out_shared_link
    out_executable_compile
    out_executable_link)
  cmake_parse_arguments(PARSE_ARGV 4 ARG
    ""
    "LANGUAGE;FORTIFY"
    ""
  )
  if(NOT ARG_LANGUAGE MATCHES "^(C|CXX)$")
    message(FATAL_ERROR
      "therock_get_hardening_command_flags requires LANGUAGE C or CXX"
    )
  endif()
  foreach(_out_var
      "${out_compile}"
      "${out_shared_link}"
      "${out_executable_compile}"
      "${out_executable_link}")
    set(${_out_var} "" PARENT_SCOPE)
  endforeach()

  _therock_validate_hardening_policy()
  if(THEROCK_HARDENING_POLICY STREQUAL "OFF" OR
      THEROCK_HARDENING_EXCLUDED)
    return()
  endif()
  _therock_hardening_validate_compiler(
    "${ARG_LANGUAGE}" "${CMAKE_${ARG_LANGUAGE}_COMPILER_ID}"
  )
  _therock_hardening_report_sanitizer()

  set(_compile_options ${_THEROCK_HARDENING_COMPILE_OPTIONS})
  # Clear inherited definitions even when the wrapper disables fortification.
  list(APPEND _compile_options ${_THEROCK_HARDENING_FORTIFY_RESET_OPTIONS})
  if(ARG_FORTIFY AND NOT THEROCK_SANITIZER)
    _therock_hardening_validate_fortify(
      "${ARG_LANGUAGE}"
      "${CMAKE_${ARG_LANGUAGE}_COMPILER_ID}"
      "${CMAKE_CURRENT_BINARY_DIR}"
    )
    list(APPEND _compile_options -Wp,-D_FORTIFY_SOURCE=3)
  endif()
  set(_link_options ${_THEROCK_HARDENING_LINK_OPTIONS})
  if(CMAKE_${ARG_LANGUAGE}_COMPILER_ID STREQUAL "Clang")
    set(_host_compile_options)
    foreach(_option IN LISTS _compile_options)
      list(APPEND _host_compile_options -Xarch_host "${_option}")
    endforeach()
    set(_compile_options ${_host_compile_options})
    set(_host_link_options)
    foreach(_option IN LISTS _link_options)
      list(APPEND _host_link_options -Xarch_host "${_option}")
    endforeach()
    set(_link_options ${_host_link_options})
    set(_pie_compile_options -Xarch_host -fPIE)
    set(_pie_link_options -Xarch_host -pie)
  else()
    set(_pie_compile_options -fPIE)
    set(_pie_link_options -pie)
  endif()

  set(_executable_compile_options
    ${_compile_options} ${_pie_compile_options}
  )
  set(_executable_link_options ${_link_options} ${_pie_link_options})
  list(JOIN _compile_options " " _compile_flags)
  list(JOIN _link_options " " _shared_link_flags)
  list(JOIN _executable_compile_options " " _executable_compile_flags)
  list(JOIN _executable_link_options " " _executable_link_flags)
  set(${out_compile} "${_compile_flags}" PARENT_SCOPE)
  set(${out_shared_link} "${_shared_link_flags}" PARENT_SCOPE)
  set(${out_executable_compile} "${_executable_compile_flags}" PARENT_SCOPE)
  set(${out_executable_link} "${_executable_link_flags}" PARENT_SCOPE)
endfunction()

# Independent nested CMake builds can use this module directly as their
# CMAKE_PROJECT_INCLUDE. The opt-in avoids scheduling a second target pass when
# the module is included normally by a generated TheRock project init file.
if(THEROCK_COMPILER_HARDENING_PROJECT_INCLUDE)
  therock_initialize_hardening_policy()
  therock_schedule_hardening()
endif()

cmake_policy(POP)
