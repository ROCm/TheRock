# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Target metadata is maintained as global properties:
#   THEROCK_AMDGPU_TARGETS: List of gfx target names
#   THEROCK_AMDGPU_TARGET_FAMILIES: List of target families (may contain duplicates)
#   THEROCK_AMDGPU_TARGET_NAME_{gfx_target}: Product name of the gfx target
#   THEROCK_AMDGPU_TARGET_FAMILY_{family}: List of gfx targets within a named
#     family
#   THEROCK_AMDGPU_PROJECT_TARGET_EXCLUDES_${project_name}: Project target keyed
#     list of gfx targets to exclude when building the target.
#   THEROCK_AMDGPU_TARGET_HW_INDEPENDENT_${gfx_target}: TRUE if the target is
#     hardware-independent (e.g. amdgcnspirv: portable SPIR-V finalized to native
#     ISA at load time by an already-installed runtime; runs on any HW).
#   THEROCK_AMDGPU_PROJECT_TARGET_INCOMPATIBLE_${project_name}: Project target
#     keyed list of gfx targets the project is fundamentally incompatible with.
#     Enabling such a project for such a target is a configuration error (FATAL).
#
# Note that each gfx_target will also create a family of the same name.
set_property(GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)

# Declares an AMDGPU target, associating it with family names and optionally
# setting additional characteristics.
# Args: gfx_target product_name
#
# Keyword Args:
# FAMILY: List of family names to associate the gfx target with.
# HW_INDEPENDENT: Option marking this target as hardware-independent (portable,
#   not tied to a specific GPU ISA; e.g. amdgcnspirv). Used to reason about which
#   components make sense for it.
# EXCLUDE_TARGET_PROJECTS: sub-project names for which this target should be
#   filtered out of GPU_TARGETS while the project still builds. This is used to
#   work around bugs during bringup and should not be set on any fully supported
#   targets.
# INCOMPATIBLE_TARGET_PROJECTS: sub-project names that are fundamentally
#   incompatible with this target (cannot be built for it by design). This does
#   NOT silently disable them; instead enabling such a project for this target is
#   reported as a fatal configuration error via therock_check_target_compatible().
#   The intent is to surface bad configurations high up rather than quietly drop
#   components.
function(therock_add_amdgpu_target gfx_target product_name)
  cmake_parse_arguments(PARSE_ARGV 2 ARG
    "HW_INDEPENDENT"
    ""
    "FAMILY;EXCLUDE_TARGET_PROJECTS;INCOMPATIBLE_TARGET_PROJECTS"
  )

  get_property(_targets GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)
  if("${gfx_target}" IN_LIST _targets)
    message(FATAL_ERROR "AMDGPU target ${gfx_target} already defined")
  endif()
  set_property(GLOBAL APPEND PROPERTY THEROCK_AMDGPU_TARGETS "${gfx_target}")
  set_property(GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_NAME_${gfx_target}" "${product_name}")
  set_property(GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_HW_INDEPENDENT_${gfx_target}" "${ARG_HW_INDEPENDENT}")
  foreach(project_name in ${ARG_EXCLUDE_TARGET_PROJECTS})
    set_property(GLOBAL APPEND PROPERTY THEROCK_AMDGPU_PROJECT_TARGET_EXCLUDES_${project_name} "${gfx_target}")
  endforeach()
  foreach(project_name IN LISTS ARG_INCOMPATIBLE_TARGET_PROJECTS)
    set_property(GLOBAL APPEND PROPERTY THEROCK_AMDGPU_PROJECT_TARGET_INCOMPATIBLE_${project_name} "${gfx_target}")
  endforeach()
  foreach(_family "${gfx_target}" ${ARG_FAMILY})
    set_property(GLOBAL APPEND PROPERTY THEROCK_AMDGPU_TARGET_FAMILIES "${_family}")
    set_property(GLOBAL APPEND PROPERTY "THEROCK_AMDGPU_TARGET_FAMILY_${_family}" "${gfx_target}")
  endforeach()
endfunction()

# Fails with a fatal error if a project is enabled for a target it is declared
# incompatible with (via INCOMPATIBLE_TARGET_PROJECTS). This is the "component
# incompatibility" gate: rather than silently disabling components low in the
# tree, callers assert compatibility explicitly and a misconfiguration is
# surfaced loudly.
#
# Semantics: fires only when a per-arch target list is actually requested
# (THEROCK_AMDGPU_TARGETS is set and not the NOTFOUND sentinel) and every
# requested target is incompatible with the project. For a mixed selection where
# at least one requested target is compatible, this does not fire (the project
# builds for the compatible subset; use EXCLUDE_TARGET_PROJECTS to strip the
# incompatible ones from its GPU_TARGETS).
function(therock_check_target_compatible project_name)
  if(NOT THEROCK_AMDGPU_TARGETS OR "${THEROCK_AMDGPU_TARGETS}" MATCHES "-NOTFOUND$")
    return()
  endif()
  get_property(_incompatible GLOBAL PROPERTY
    "THEROCK_AMDGPU_PROJECT_TARGET_INCOMPATIBLE_${project_name}")
  if(NOT _incompatible)
    return()
  endif()
  set(_remaining ${THEROCK_AMDGPU_TARGETS})
  list(REMOVE_ITEM _remaining ${_incompatible})
  if(NOT _remaining)
    string(JOIN " " _targets_pretty ${THEROCK_AMDGPU_TARGETS})
    message(FATAL_ERROR
      "Project '${project_name}' is enabled but is incompatible with the "
      "selected AMDGPU target(s): ${_targets_pretty}. This component cannot be "
      "built for these targets by design. Disable it (do not include it in the "
      "enabled feature set) or select a compatible target.")
  endif()
endfunction()

# TRUE when 'project_name' is incompatible with ALL currently requested per-arch
# targets. Lets a caller prune incompatible members while expanding a feature
# group (e.g. ENABLE_ALL) without triggering the fatal check. FALSE for
# generic/no-target stages and ordinary gfx builds.
function(therock_project_incompatible out_var project_name)
  set(_result FALSE)
  if(THEROCK_AMDGPU_TARGETS AND NOT "${THEROCK_AMDGPU_TARGETS}" MATCHES "-NOTFOUND$")
    get_property(_incompatible GLOBAL PROPERTY
      "THEROCK_AMDGPU_PROJECT_TARGET_INCOMPATIBLE_${project_name}")
    if(_incompatible)
      set(_remaining ${THEROCK_AMDGPU_TARGETS})
      list(REMOVE_ITEM _remaining ${_incompatible})
      if(NOT _remaining)
        set(_result TRUE)
      endif()
    endif()
  endif()
  set("${out_var}" "${_result}" PARENT_SCOPE)
endfunction()

# gfx900
therock_add_amdgpu_target(gfx900 "Vega 10 / MI25" FAMILY dgpu-all gfx900-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx90c
therock_add_amdgpu_target(gfx90c "AMD Renoir/Lucienne/Cezanne iGPU" FAMILY igpu-all gfx90c-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx906 (separate family - different instruction support from gfx908/gfx90a)
therock_add_amdgpu_target(gfx906 "Radeon VII / MI50 CDNA" FAMILY dgpu-all gfx906-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx908 (separate family - different instruction support from gfx906/gfx90a)
therock_add_amdgpu_target(gfx908 "MI100 CDNA" FAMILY dcgpu-all gfx908-dcgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
)

# gfx90a (separate family - different instruction support from gfx906/gfx908)
therock_add_amdgpu_target(gfx90a "MI210/250 CDNA" FAMILY dcgpu-all gfx90a-dcgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
)

# gfx94X family
therock_add_amdgpu_target(gfx942 "MI300A/MI300X CDNA" FAMILY dcgpu-all gfx94X-all gfx94X-dcgpu)

# gfx950
therock_add_amdgpu_target(gfx950 "MI350X/MI355X CDNA" FAMILY dcgpu-all gfx950-all gfx950-dcgpu)

# gfx101X family
therock_add_amdgpu_target(gfx1010 "AMD RX 5700" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1011 "AMD Radeon Pro V520" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

therock_add_amdgpu_target(gfx1012 "AMD RX 5500" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx103X family
therock_add_amdgpu_target(gfx1030 "AMD RX 6800 / XT" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/4836
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1031 "AMD RX 6700 / XT" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/4836
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1032 "AMD RX 6600" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/4836
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1033 "AMD Van Gogh iGPU" FAMILY igpu-all gfx103X-all gfx103X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
    composable_kernel
)
therock_add_amdgpu_target(gfx1034 "AMD RX 6500 XT" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/4836
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1035 "AMD Radeon 680M Laptop iGPU" FAMILY igpu-all gfx103X-all gfx103X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/4836
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1036 "AMD Raphael iGPU" FAMILY igpu-all gfx103X-all gfx103X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/4836
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
    hipTensor # https://github.com/ROCm/TheRock/issues/2074
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx110X family
therock_add_amdgpu_target(gfx1100 "AMD RX 7900 XTX" FAMILY dgpu-all gfx110X-all gfx110X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1101 "AMD RX 7800 XT" FAMILY dgpu-all gfx110X-all gfx110X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1102 "AMD RX 7700S/Framework Laptop 16" FAMILY dgpu-all gfx110X-all gfx110X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1103 "AMD Radeon 780M Laptop iGPU" FAMILY igpu-all gfx110X-all gfx110X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl # https://github.com/ROCm/TheRock/issues/150
    rccl-tests
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx115X family
therock_add_amdgpu_target(gfx1150 "AMD Strix Point iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl # https://github.com/ROCm/TheRock/issues/150
    rccl-tests
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1151 "AMD Strix Halo iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1152 "AMD Krackan 1 iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl # https://github.com/ROCm/TheRock/issues/150
    rccl-tests
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1153 "AMD Radeon 820M iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl # https://github.com/ROCm/TheRock/issues/150
    rccl-tests
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx120X family
therock_add_amdgpu_target(gfx1200 "AMD RX 9060 / XT" FAMILY dgpu-all gfx120X-all
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)
therock_add_amdgpu_target(gfx1201 "AMD RX 9070 / XT" FAMILY dgpu-all gfx120X-all
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/2892
)

# gfx125X family
therock_add_amdgpu_target(gfx1250 "AMD Instinct MI450/MI450X/MI455X CDNA" FAMILY dcgpu-all gfx125X-all gfx125X-dcgpu)

# amdgcnspirv: hardware-independent portable SPIR-V target. Runs on any HW where
# a ROCm runtime is already installed (the runtime finalizes SPIR-V to native ISA
# at load time). Projects with GPU-specific components that do not yet support
# SPIR-V are declared INCOMPATIBLE: enabling any of them for this target is a
# fatal configuration error (see therock_check_target_compatible), rather than
# silently disabling them.
therock_add_amdgpu_target(amdgcnspirv "AMDGPU portable SPIR-V" FAMILY gpu-generic
  HW_INDEPENDENT
  INCOMPATIBLE_TARGET_PROJECTS
    MIOpen # https://github.com/ROCm/TheRock/issues/6918
    ROCR-Runtime # https://github.com/ROCm/TheRock/issues/6918
    aqlprofile # https://github.com/ROCm/TheRock/issues/6918
    composable_kernel # https://github.com/ROCm/TheRock/issues/6918
    hip-clr # https://github.com/ROCm/TheRock/issues/6918
    hip-tests # https://github.com/ROCm/TheRock/issues/6918
    hipBLAS # https://github.com/ROCm/TheRock/issues/6918
    hipBLAS-common # https://github.com/ROCm/TheRock/issues/6918
    hipBLASLt # https://github.com/ROCm/TheRock/issues/6918
    hipCUB # https://github.com/ROCm/TheRock/issues/6918
    hipDNN # https://github.com/ROCm/TheRock/issues/6918
    hipDNN_samples # https://github.com/ROCm/TheRock/issues/6918
    hipFFT # https://github.com/ROCm/TheRock/issues/6918
    hipInfo # https://github.com/ROCm/TheRock/issues/6918
    hipRAND # https://github.com/ROCm/TheRock/issues/6918
    hipSOLVER # https://github.com/ROCm/TheRock/issues/6918
    hipSPARSE # https://github.com/ROCm/TheRock/issues/6918
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/6918
    hipTensor # https://github.com/ROCm/TheRock/issues/6918
    hipblasltprovider # https://github.com/ROCm/TheRock/issues/6918
    hipdnn_integration_tests # https://github.com/ROCm/TheRock/issues/6918
    hipfile # https://github.com/ROCm/TheRock/issues/6918
    hipkernelprovider # https://github.com/ROCm/TheRock/issues/6918
    libhipcxx # https://github.com/ROCm/TheRock/issues/6918
    miopenprovider # https://github.com/ROCm/TheRock/issues/6918
    mirage # https://github.com/ROCm/TheRock/issues/6918
    mxDataGenerator # https://github.com/ROCm/TheRock/issues/6918
    ocl-clr # https://github.com/ROCm/TheRock/issues/6918
    rccl # https://github.com/ROCm/TheRock/issues/6918
    rccl-tests # https://github.com/ROCm/TheRock/issues/6918
    rdc # https://github.com/ROCm/TheRock/issues/6918
    rocALUTION # https://github.com/ROCm/TheRock/issues/6918
    rocBLAS # https://github.com/ROCm/TheRock/issues/6918
    rocFFT # https://github.com/ROCm/TheRock/issues/6918
    rocPRIM # https://github.com/ROCm/TheRock/issues/6918
    rocPRIM_tests # https://github.com/ROCm/TheRock/issues/6918
    rocRAND # https://github.com/ROCm/TheRock/issues/6918
    rocRoller # https://github.com/ROCm/TheRock/issues/6918
    rocSOLVER # https://github.com/ROCm/TheRock/issues/6918
    rocSPARSE # https://github.com/ROCm/TheRock/issues/6918
    rocThrust # https://github.com/ROCm/TheRock/issues/6918
    rocWMMA # https://github.com/ROCm/TheRock/issues/6918
    rocdecode # https://github.com/ROCm/TheRock/issues/6918
    rocjitsu # https://github.com/ROCm/TheRock/issues/6918
    rocjpeg # https://github.com/ROCm/TheRock/issues/6918
    rocm-kpack # https://github.com/ROCm/TheRock/issues/6918
    rocprofiler-compute # https://github.com/ROCm/TheRock/issues/6918
    rocprofiler-sdk # https://github.com/ROCm/TheRock/issues/6918
    rocprofiler-systems # https://github.com/ROCm/TheRock/issues/6918
    rocprofiler-systems-examples # https://github.com/ROCm/TheRock/issues/6918
    rocr-debug-agent-tests # https://github.com/ROCm/TheRock/issues/6918
    rocrtst # https://github.com/ROCm/TheRock/issues/6918
    rocshmem # https://github.com/ROCm/TheRock/issues/6918
    roctracer # https://github.com/ROCm/TheRock/issues/6918
)

# Optional extension targets (used for out of tree target development).
include(therock_custom_amdgpu_targets OPTIONAL)

# Validates and normalizes AMDGPU target selection cache variables.
#
# This function handles three separate target lists:
#   THEROCK_AMDGPU_TARGETS: Per-architecture targets for architecture-specific builds
#   THEROCK_DIST_AMDGPU_TARGETS: Distribution targets (used by runtime components
#     that embed device code for user-selected architectures). Controls what is
#     reported in dist_info.json and consumed by downstream tools (e.g. PyTorch
#     via `rocm-sdk targets`). Defaults to THEROCK_AMDGPU_FAMILIES.
#   THEROCK_TEST_AMDGPU_TARGETS: Targets for test artifacts marked TARGET_NEUTRAL.
#     Defaults to ALL available (registered) targets so that a single _generic
#     test artifact works on any architecture, making upload races in classic CI
#     harmless. Does NOT affect dist_info.json.
#
# In multi-arch CI, generic stages (those building architecture-independent code)
# may have no per-arch targets but still need dist targets. In this case,
# THEROCK_AMDGPU_TARGETS is set to a sentinel value "THEROCK_AMDGPU_TARGETS-NOTFOUND"
# and the error is deferred until a subproject actually needs per-arch targets.
function(therock_validate_amdgpu_targets)
  message(STATUS "Configured AMDGPU Targets:")
  string(APPEND CMAKE_MESSAGE_INDENT "  ")
  set(_expanded_targets)
  set(_explicit_selections)
  get_property(_available_families GLOBAL PROPERTY THEROCK_AMDGPU_TARGET_FAMILIES)
  list(REMOVE_DUPLICATES _available_families)
  get_property(_available_targets GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)

  # Expand per-arch families (THEROCK_AMDGPU_FAMILIES -> THEROCK_AMDGPU_TARGETS).
  foreach(_family ${THEROCK_AMDGPU_FAMILIES})
    list(APPEND _explicit_selections "${_family}")
    if(NOT "${_family}" IN_LIST _available_families)
      string(JOIN " " _families_pretty ${_available_families})
      message(FATAL_ERROR
        "THEROCK_AMDGPU_FAMILIES value '${_family}' unknown. Available: "
        ${_families_pretty})
    endif()
    get_property(_family_targets GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_FAMILY_${_family}")
    list(APPEND _expanded_targets ${_family_targets})
  endforeach()

  # And expand loose targets.
  foreach(_target ${THEROCK_AMDGPU_TARGETS})
    list(APPEND _explicit_selections "${_target}")
    list(APPEND _expanded_targets ${_target})
  endforeach()

  # Validate per-arch targets.
  list(REMOVE_DUPLICATES _expanded_targets)
  foreach(_target ${_expanded_targets})
    string(JOIN " " _targets_pretty ${_available_targets})
    if(NOT "${_target}" IN_LIST _available_targets)
      message(FATAL_ERROR "Unknown AMDGPU target '${_target}'. Available: "
        ${_targets_pretty})
    endif()
    get_property(_target_name GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_NAME_${_target}")
    message(STATUS "* ${_target} : ${_target_name}")
  endforeach()

  # Expand dist families (THEROCK_DIST_AMDGPU_FAMILIES -> THEROCK_DIST_AMDGPU_TARGETS).
  # If neither THEROCK_DIST_AMDGPU_FAMILIES nor THEROCK_DIST_AMDGPU_TARGETS is set,
  # dist defaults to the build families (THEROCK_AMDGPU_FAMILIES).
  set(_dist_expanded_targets "${THEROCK_DIST_AMDGPU_TARGETS}")
  set(_dist_families "${THEROCK_DIST_AMDGPU_FAMILIES}")
  if(NOT _dist_families AND NOT _dist_expanded_targets)
    set(_dist_families "${THEROCK_AMDGPU_FAMILIES}")
  endif()
  foreach(_family ${_dist_families})
    if(NOT "${_family}" IN_LIST _available_families)
      string(JOIN " " _families_pretty ${_available_families})
      message(FATAL_ERROR
        "THEROCK_DIST_AMDGPU_FAMILIES value '${_family}' unknown. Available: "
        ${_families_pretty})
    endif()
    get_property(_family_targets GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_FAMILY_${_family}")
    list(APPEND _dist_expanded_targets ${_family_targets})
  endforeach()
  list(REMOVE_DUPLICATES _dist_expanded_targets)

  # Report dist targets if different from per-arch targets.
  if(_dist_expanded_targets AND NOT "${_dist_expanded_targets}" STREQUAL "${_expanded_targets}")
    message(STATUS "Dist targets: ${_dist_expanded_targets}")
  endif()

  # Expand test families (THEROCK_TEST_AMDGPU_FAMILIES -> THEROCK_TEST_AMDGPU_TARGETS).
  # If neither THEROCK_TEST_AMDGPU_FAMILIES nor THEROCK_TEST_AMDGPU_TARGETS is set,
  # test targets default to ALL available targets so that a single _generic test
  # artifact can be downloaded and run on any architecture, making the classic CI
  # upload race harmless.
  set(_test_families "${THEROCK_TEST_AMDGPU_FAMILIES}")
  set(_test_expanded_targets "${THEROCK_TEST_AMDGPU_TARGETS}")
  if(NOT _test_families AND NOT _test_expanded_targets)
    set(_test_expanded_targets "${_available_targets}")
  else()
    foreach(_family ${_test_families})
      if(NOT "${_family}" IN_LIST _available_families)
        string(JOIN " " _families_pretty ${_available_families})
        message(FATAL_ERROR
          "THEROCK_TEST_AMDGPU_FAMILIES value '${_family}' unknown. Available: "
          ${_families_pretty})
      endif()
      get_property(_family_targets GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_FAMILY_${_family}")
      list(APPEND _test_expanded_targets ${_family_targets})
    endforeach()
    list(REMOVE_DUPLICATES _test_expanded_targets)
  endif()

  # Report test targets if different from per-arch targets.
  if(_test_expanded_targets AND NOT "${_test_expanded_targets}" STREQUAL "${_expanded_targets}")
    message(STATUS "Test targets: ${_test_expanded_targets}")
  endif()

  # Handle the case where per-arch targets are empty but dist targets exist.
  # This is valid for generic stages in multi-arch CI that don't build
  # architecture-specific code but need to know about all dist targets.
  if(NOT _expanded_targets AND _dist_expanded_targets)
    message(STATUS "(No per-arch targets - generic stage using dist targets only)")
    set(THEROCK_AMDGPU_TARGETS "THEROCK_AMDGPU_TARGETS-NOTFOUND" PARENT_SCOPE)
    set(THEROCK_AMDGPU_TARGETS_SPACES "" PARENT_SCOPE)
  elseif(NOT _expanded_targets AND NOT _dist_expanded_targets)
    message(FATAL_ERROR
      "No AMDGPU target selected: make a selection via THEROCK_AMDGPU_FAMILIES "
      "or THEROCK_AMDGPU_TARGETS (or THEROCK_DIST_AMDGPU_FAMILIES for dist-only)."
    )
  else()
    # Export per-arch targets to parent scope.
    set(THEROCK_AMDGPU_TARGETS "${_expanded_targets}" PARENT_SCOPE)
    string(JOIN " " _expanded_targets_spaces ${_expanded_targets})
    set(THEROCK_AMDGPU_TARGETS_SPACES "${_expanded_targets_spaces}" PARENT_SCOPE)
  endif()

  # Export dist targets to parent scope.
  if(_dist_expanded_targets)
    set(THEROCK_DIST_AMDGPU_TARGETS "${_dist_expanded_targets}" PARENT_SCOPE)
    string(JOIN " " _dist_expanded_targets_spaces ${_dist_expanded_targets})
    set(THEROCK_DIST_AMDGPU_TARGETS_SPACES "${_dist_expanded_targets_spaces}" PARENT_SCOPE)
  else()
    set(THEROCK_DIST_AMDGPU_TARGETS "THEROCK_DIST_AMDGPU_TARGETS-NOTFOUND" PARENT_SCOPE)
    set(THEROCK_DIST_AMDGPU_TARGETS_SPACES "" PARENT_SCOPE)
  endif()

  # Export test targets to parent scope.
  if(_test_expanded_targets)
    set(THEROCK_TEST_AMDGPU_TARGETS "${_test_expanded_targets}" PARENT_SCOPE)
    string(JOIN " " _test_expanded_targets_spaces ${_test_expanded_targets})
    set(THEROCK_TEST_AMDGPU_TARGETS_SPACES "${_test_expanded_targets_spaces}" PARENT_SCOPE)
  else()
    set(THEROCK_TEST_AMDGPU_TARGETS "THEROCK_TEST_AMDGPU_TARGETS-NOTFOUND" PARENT_SCOPE)
    set(THEROCK_TEST_AMDGPU_TARGETS_SPACES "" PARENT_SCOPE)
  endif()

  if(NOT THEROCK_AMDGPU_DIST_BUNDLE_NAME)
    list(LENGTH _explicit_selections _explicit_count)
    if(_explicit_count GREATER "1")
      message(FATAL_ERROR
        "More than one AMDGPU target bundle selected (${_explicit_selections}): "
        "THEROCK_AMDGPU_DIST_BUNDLE_NAME must be set explicitly since it cannot "
        "be inferred."
      )
    endif()
    set(THEROCK_AMDGPU_DIST_BUNDLE_NAME "${_explicit_selections}" PARENT_SCOPE)
    if(_explicit_selections)
      message(STATUS "* Dist bundle: ${_explicit_selections}")
    endif()
  else()
    message(STATUS "* Dist bundle: ${THEROCK_AMDGPU_DIST_BUNDLE_NAME}")
  endif()
endfunction()

function(therock_get_amdgpu_target_name out_var gfx_target)
  get_property(_name GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_NAME_${gfx_target}")
  set("${out_var}" "${_name}" PARENT_SCOPE)
endfunction()
