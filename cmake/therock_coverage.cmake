# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Code coverage instrumentation for sub-projects.
#
# Coverage has to be scoped to the component being measured. Instrumented code
# emits .profraw files whenever it runs, so instrumenting a dependency of the
# component under test folds that dependency's counters into the report and
# makes the coverage denominator move whenever the dependency changes. A single
# shared option name (such as the BUILD_CODE_COVERAGE that rocm-libraries
# projects define today) cannot express that scoping: every sub-project reading
# it would be instrumented as well. Coverage is therefore requested per project,
# through an option named after the project's logical target name in upper case
# (hipDNN -> HIPDNN_ENABLE_COVERAGE).
#
# Three request mechanisms are supported, in decreasing order of precedence:
#
#   1. -D<PROJECT>_ENABLE_COVERAGE=ON|OFF
#      Direct per-project request, for local builds of a known project. An
#      explicit OFF also opts one project out of the group flags below.
#
#   2. -DTHEROCK_COVERAGE_PROJECTS="hiprand;rocFFT"
#      Case-insensitive project list, so that CI can forward the project names
#      it detected as changed without knowing how each flag is spelled. Accepts
#      a comma- or semicolon-separated list plus the values "all" and "none".
#
#   3. -DTHEROCK_COVERAGE_ROCM_LIBRARIES_ALL=ON
#      -DTHEROCK_COVERAGE_ROCM_SYSTEMS_ALL=ON
#      -DTHEROCK_COVERAGE_ALL=ON
#      Instrument every component of a monorepo. Nightly coverage runs use
#      these to build one instrumented stack and then measure each component
#      separately. The two monorepo flags are independent so that a nightly run
#      can instrument the libraries without paying to instrument the whole ROCm
#      system stack; THEROCK_COVERAGE_ALL enables both.
#
# See docs/development/code_coverage.md.

option(THEROCK_COVERAGE_ALL
  "Enable code coverage instrumentation for all rocm-libraries and rocm-systems components" OFF)
option(THEROCK_COVERAGE_ROCM_LIBRARIES_ALL
  "Enable code coverage instrumentation for all rocm-libraries components (math-libs, ml-libs, cv-libs, ...)" OFF)
option(THEROCK_COVERAGE_ROCM_SYSTEMS_ALL
  "Enable code coverage instrumentation for all rocm-systems components (base, core, profiler, ...)" OFF)
set(THEROCK_COVERAGE_PROJECTS "" CACHE STRING
  "Case-insensitive list of projects to instrument for code coverage (also accepts 'all' or 'none')")

# therock_coverage_init
# Expands the coarse coverage requests into the per-project form that
# therock_coverage_get_subproject_args consumes. Must be called from the
# top-level CMakeLists.txt before any sub-project is declared.
function(therock_coverage_init)
  # Normalize the project list. CI passes whatever casing its change detection
  # produced ("hiprand", "hipRAND", "HIPRAND"), so fold to upper case here
  # rather than making every caller match the target's casing.
  set(_projects "${THEROCK_COVERAGE_PROJECTS}")
  string(REPLACE "," ";" _projects "${_projects}")
  set(_enabled_projects)
  foreach(_project IN LISTS _projects)
    string(STRIP "${_project}" _project)
    if(NOT _project)
      continue()
    endif()
    string(TOUPPER "${_project}" _project)
    if(_project STREQUAL "NONE")
      continue()
    elseif(_project STREQUAL "ALL")
      set(THEROCK_COVERAGE_ALL ON)
      set(THEROCK_COVERAGE_ALL ON PARENT_SCOPE)
      continue()
    endif()
    set("${_project}_ENABLE_COVERAGE" ON PARENT_SCOPE)
    list(APPEND _enabled_projects "${_project}")
  endforeach()

  # THEROCK_COVERAGE_ALL is defined as "both monorepos", not "every sub-project
  # in the super-build": instrumenting amd-llvm or the bundled third-party
  # dependencies costs build time and contributes nothing to a component report.
  if(THEROCK_COVERAGE_ALL)
    set(THEROCK_COVERAGE_ROCM_LIBRARIES_ALL ON PARENT_SCOPE)
    set(THEROCK_COVERAGE_ROCM_SYSTEMS_ALL ON PARENT_SCOPE)
  endif()

  if(_enabled_projects OR THEROCK_COVERAGE_ALL
     OR THEROCK_COVERAGE_ROCM_LIBRARIES_ALL OR THEROCK_COVERAGE_ROCM_SYSTEMS_ALL)
    message(STATUS "Code coverage instrumentation enabled:")
    if(_enabled_projects)
      message(STATUS "  projects: ${_enabled_projects}")
    endif()
    if(THEROCK_COVERAGE_ALL OR THEROCK_COVERAGE_ROCM_LIBRARIES_ALL)
      message(STATUS "  all rocm-libraries components")
    endif()
    if(THEROCK_COVERAGE_ALL OR THEROCK_COVERAGE_ROCM_SYSTEMS_ALL)
      message(STATUS "  all rocm-systems components")
    endif()
  endif()
endfunction()

# therock_coverage_source_group
# Reports which monorepo a sub-project's sources come from: "rocm-libraries",
# "rocm-systems", or empty for in-tree and third-party sources. There is no
# declared grouping to consult, but a component's monorepo is exactly what its
# EXTERNAL_SOURCE_DIR points into.
function(therock_coverage_source_group out_var external_source_dir)
  set("${out_var}" "" PARENT_SCOPE)
  if(NOT external_source_dir)
    return()
  endif()
  if(THEROCK_ROCM_LIBRARIES_SOURCE_DIR)
    cmake_path(IS_PREFIX THEROCK_ROCM_LIBRARIES_SOURCE_DIR "${external_source_dir}"
      NORMALIZE _is_rocm_libraries)
    if(_is_rocm_libraries)
      set("${out_var}" "rocm-libraries" PARENT_SCOPE)
      return()
    endif()
  endif()
  if(THEROCK_ROCM_SYSTEMS_SOURCE_DIR)
    cmake_path(IS_PREFIX THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${external_source_dir}"
      NORMALIZE _is_rocm_systems)
    if(_is_rocm_systems)
      set("${out_var}" "rocm-systems" PARENT_SCOPE)
    endif()
  endif()
endfunction()

# therock_coverage_get_subproject_args
# Sets out_var to the coverage options to add to a sub-project's configure
# command line, or to an empty list when the sub-project is not being measured.
function(therock_coverage_get_subproject_args
    out_var
    logical_target_name
    external_source_dir)
  set("${out_var}" "" PARENT_SCOPE)

  string(TOUPPER "${logical_target_name}" _project_name)
  set(_coverage_var_name "${_project_name}_ENABLE_COVERAGE")

  if(DEFINED ${_coverage_var_name})
    # An explicit request wins over the group flags, in both directions, so that
    # a nightly run can instrument a whole monorepo except for one component.
    set(_enabled "${${_coverage_var_name}}")
  else()
    set(_enabled OFF)
    therock_coverage_source_group(_group "${external_source_dir}")
    if(THEROCK_COVERAGE_ROCM_LIBRARIES_ALL AND _group STREQUAL "rocm-libraries")
      set(_enabled ON)
    elseif(THEROCK_COVERAGE_ROCM_SYSTEMS_ALL AND _group STREQUAL "rocm-systems")
      set(_enabled ON)
    endif()
  endif()

  if(NOT _enabled)
    return()
  endif()

  # The project-specific option from the RFC, which components are moving to.
  set(_args "-D${_coverage_var_name}=ON")

  # Compatibility with the option names components define today, for example
  # BUILD_CODE_COVERAGE in hipRAND and CODE_COVERAGE in rocRAND. Passing them
  # here is safe precisely because these arguments only reach the configure
  # command of the sub-project being measured, so they cannot instrument a
  # dependency the way a super-build-wide -DBUILD_CODE_COVERAGE would. Drop
  # these once every component accepts <PROJECT>_ENABLE_COVERAGE.
  list(APPEND _args "-DBUILD_CODE_COVERAGE=ON" "-DCODE_COVERAGE=ON")

  message(STATUS "  ENABLE CODE COVERAGE: ${logical_target_name}")
  set("${out_var}" "${_args}" PARENT_SCOPE)
endfunction()
