# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_coverage_configure
# Resolves TheRock's -D<PROJECT>_ENABLE_COVERAGE for one subproject.
#
# Outputs:
#   out_init_contents: Lines to append to the subproject's init file.
#   out_cmake_arg: "-D<upstream option>=<value>" to pass to the subproject, or
#     empty.
#   out_project_include: Generated file to pass as CMAKE_PROJECT_INCLUDE, or
#     empty.
#
# The registry lookup and its error come first, so that a typo or a project
# with no registered option fails configure without first appending linker
# flags and rewriting this subproject's compile rules.
#
# CMake definitions are conventionally upper case, so only the upper case
# spelling of TheRock's flag is honored (i.e. HIPDNN_ENABLE_COVERAGE, not
# hipDNN_ENABLE_COVERAGE).
#
# The option names upstream are not standardised: most projects take
# BUILD_CODE_COVERAGE or CODE_COVERAGE, a few take <PROJECT>_ENABLE_COVERAGE,
# RCCL takes ENABLE_CODE_COVERAGE. The generic ones are the reason the
# translation is per subproject rather than a global: defining
# BUILD_CODE_COVERAGE for the whole build would instrument every project that
# happens to understand it, which is exactly what per-project coverage is
# trying to avoid.
#
# DEFINED rather than the value, because an explicit =OFF naming a project
# with no registered option is still a mistake worth reporting, and is a
# spelling a typo readily takes.
function(therock_coverage_configure
    out_init_contents
    out_cmake_arg
    out_project_include
    logical_target_name
    compiler_toolchain
    coverage_file)
  set("${out_init_contents}" "" PARENT_SCOPE)
  set("${out_cmake_arg}" "" PARENT_SCOPE)
  set("${out_project_include}" "" PARENT_SCOPE)

  string(TOUPPER "${logical_target_name}" _project_name)
  set(_var_name "${_project_name}_ENABLE_COVERAGE")
  if(NOT DEFINED ${_var_name})
    return()
  endif()

  set(_option "${THEROCK_COVERAGE_OPTION_${_project_name}}")
  if(NOT _option)
    message(FATAL_ERROR
      "${logical_target_name}: ${_var_name} was set but the project "
      "has no coverage option registered. Add one to COVERAGE_PROJECTS in "
      "build_tools/github_actions/configure_coverage_ci.py.")
  endif()

  # The super-project may be configured with MSVC while the subproject is
  # compiled with clang from amd-llvm, as most ROCm subprojects are on Windows.
  # The device compilation negation below is clang specific, and handing a
  # subproject compiled by MSVC itself its coverage option without it would
  # instrument it halfway, so coverage is a no-op there rather than a partial
  # one.
  if(MSVC AND NOT compiler_toolchain MATCHES "^amd-(llvm|hip)$")
    if(${_var_name})
      message(WARNING
        "${logical_target_name}: ${_var_name} is ignored for subprojects "
        "compiled with MSVC. Coverage instrumentation needs the amd-llvm "
        "clang toolchain.")
    endif()
    return()
  endif()

  set("${out_cmake_arg}" "-D${_option}=${${_var_name}}" PARENT_SCOPE)
  if(NOT ${_var_name})
    return()
  endif()

  # Give the LLVM profile runtime the libraries it depends on.
  #
  # -fprofile-instr-generate links libclang_rt.profile_rocm.a, whose
  # InstrProfilingPlatformROCm.cpp calls dlsym, dladdr and pthread_once. The
  # driver does not put -ldl or -lpthread on the link line, and on the
  # manylinux base those symbols are still in libdl and libpthread rather than
  # in libc, so an instrumented shared library is left with undefined
  # references. The library itself links -- shared objects tolerate that -- but
  # every executable linking against it then fails
  # --no-allow-shlib-undefined, which is how this shows up: hipRAND's tests
  # cannot link against an instrumented libhiprand.so.
  #
  # push-state/pop-state keeps --no-as-needed scoped to these two, so the
  # DT_NEEDED entries are recorded no matter where the flags land relative to
  # the objects that need them, without forcing a DT_NEEDED for every other
  # library on the line. Neither library nor the GNU-style flags exist for
  # lld-link on Windows.
  set(_init_contents)
  if(NOT WIN32)
    set(_link_flags
      " -Wl,--push-state,--no-as-needed -ldl -lpthread -Wl,--pop-state")
    foreach(_kind IN ITEMS EXE SHARED MODULE)
      string(APPEND _init_contents
        "string(APPEND CMAKE_${_kind}_LINKER_FLAGS \"${_link_flags}\")\n")
    endforeach()
  endif()
  set("${out_init_contents}" "${_init_contents}" PARENT_SCOPE)

  # Keep the instrumentation off the GPU.
  #
  # The upstream coverage options pass -fprofile-instr-generate and
  # -fcoverage-mapping unqualified, and the HIP driver forwards both to the
  # device compilation, so kernels are instrumented too. That is not what this
  # pipeline measures, and it does not survive contact with the runtime: the
  # device image gains a per-TU __llvm_profile_sections_ symbol that
  # libclang_rt.profile_rocm.a looks up with hipModuleGetGlobal at exit, and a
  # __profd_ for every kernel. Symbols the runtime cannot resolve abort the
  # process after the tests have already passed (rocSPARSE, on a rocPRIM
  # trampoline kernel), and where the readback does succeed it writes a second
  # profile named after the device -- gfx942:sramecc+:xnack-.0.<host name> --
  # which cannot be uploaded as an artifact because of the colons.
  #
  # -Xarch_device confines the negation to the device compilation; the host
  # keeps -fprofile-instrument=clang and its coverage mapping. Order decides
  # this, since the last of a -f/-fno- pair wins and the project's own flags
  # come from COMPILE_OPTIONS, which land after CMAKE_<LANG>_FLAGS. Appending
  # to the compile rule after <FLAGS> is the one position that is reliably
  # last. The rule does not exist yet in CMAKE_PROJECT_TOP_LEVEL_INCLUDES,
  # which runs before any language is enabled, hence the separate file below.
  #
  # This reaches the languages a project() call enables, which is where the
  # HIP sources in these projects are compiled -- as CXX, through the amd-hip
  # toolchain. A language brought up later by a bare enable_language() is not
  # covered, and would need the same rewrite from wherever that happens.
  set(_host_only_contents
    "# Generated by therock_coverage.cmake; see the coverage comment there.\n")
  # Runs on every project() call, so the rewrite has to be idempotent.
  # -Wno-unused-command-line-argument because -Xarch_device is unused for
  # the plain C++ sources that make up most of these projects, and several
  # of them build with -Werror.
  foreach(_lang IN ITEMS C CXX HIP)
    string(APPEND _host_only_contents
      "if(CMAKE_${_lang}_COMPILE_OBJECT AND NOT CMAKE_${_lang}_COMPILE_OBJECT MATCHES \"fno-profile-instr-generate\")\n"
      "  string(REPLACE \"<FLAGS>\" \"<FLAGS> -Xarch_device -fno-profile-instr-generate -Xarch_device -fno-coverage-mapping -Wno-unused-command-line-argument\"\n"
      "    CMAKE_${_lang}_COMPILE_OBJECT \"\${CMAKE_${_lang}_COMPILE_OBJECT}\")\n"
      "endif()\n")
  endforeach()
  file(CONFIGURE OUTPUT "${coverage_file}"
    CONTENT "${_host_only_contents}" @ONLY ESCAPE_QUOTES)
  set("${out_project_include}" "${coverage_file}" PARENT_SCOPE)
endfunction()
