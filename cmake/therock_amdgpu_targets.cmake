# Target metadata is maintained as global properties:
#   THEROCK_AMDGPU_TARGETS: List of gfx target names
#   THEROCK_AMDGPU_TARGET_FAMILIES: List of target families (may contain duplicates)
#   THEROCK_AMDGPU_TARGET_NAME_{gfx_target}: Product name of the gfx target
#   THEROCK_AMDGPU_TARGET_FAMILY_{family}: List of gfx targets within a named
#     family
#   THEROCK_AMDGPU_PROJECT_TARGET_EXCLUDES_${project_name}: Project target keyed
#     list of gfx targets to exclude when building the target.
#
# Note that each gfx_target will also create a family of the same name.
set_property(GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)

# Declares an AMDGPU target, associating it with family names and optionally
# setting additional characteristics.
# Args: gfx_target product_name
#
# Keyword Args:
# FAMILY: List of family names to associate the gfx target with.
# EXCLUDE_TARGET_PROJECTS: sub-project names for which this target should be
#   filtered out. This is used to work around bugs during bringup and should
#   not be set on any fully supported targets.
function(therock_add_amdgpu_target gfx_target product_name)
  cmake_parse_arguments(PARSE_ARGV 2 ARG
    ""
    ""
    "FAMILY;EXCLUDE_TARGET_PROJECTS"
  )

  get_property(_targets GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)
  if("${gfx_target}" IN_LIST _targets)
    message(FATAL_ERROR "AMDGPU target ${gfx_target} already defined")
  endif()
  set_property(GLOBAL APPEND PROPERTY THEROCK_AMDGPU_TARGETS "${gfx_target}")
  set_property(GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_NAME_${gfx_target}" "${product_name}")
  foreach(project_name in ${ARG_EXCLUDE_TARGET_PROJECTS})
    set_property(GLOBAL APPEND PROPERTY THEROCK_AMDGPU_PROJECT_TARGET_EXCLUDES_${project_name} "${gfx_target}")
  endforeach()
  foreach(_family "${gfx_target}" ${ARG_FAMILY})
    set_property(GLOBAL APPEND PROPERTY THEROCK_AMDGPU_TARGET_FAMILIES "${_family}")
    set_property(GLOBAL APPEND PROPERTY "THEROCK_AMDGPU_TARGET_FAMILY_${_family}" "${gfx_target}")
  endforeach()
endfunction()

# gfx90X family
therock_add_amdgpu_target(gfx906 "Radeon VII / MI50 CDNA" FAMILY dgpu-all gfx90X-all gfx90X-dgpu gfx90X-dcgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)
therock_add_amdgpu_target(gfx908 "MI100 CDNA" FAMILY gfx90X-all dcgpu-all gfx90X-dcgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
)
therock_add_amdgpu_target(gfx90a "MI210/250 CDNA" FAMILY gfx90X-all dcgpu-all gfx90X-dcgpu
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
)
therock_add_amdgpu_target(gfx1011 "AMD Radeon Pro V520" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)
therock_add_amdgpu_target(gfx1012 "AMD RX 5500" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)

# gfx103X family
therock_add_amdgpu_target(gfx1030 "AMD RX 6800 / XT" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)
therock_add_amdgpu_target(gfx1032 "AMD RX 6600" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)
therock_add_amdgpu_target(gfx1035 "AMD Radeon 680M Laptop iGPU" igpu-all FAMILY gfx103X-all gfx103X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)
therock_add_amdgpu_target(gfx1036 "AMD Raphael iGPU" FAMILY igpu-all gfx103X-all gfx103X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)

# gfx110X family
therock_add_amdgpu_target(gfx1100 "AMD RX 7900 XTX" FAMILY dgpu-all gfx110X-all gfx110X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
)
therock_add_amdgpu_target(gfx1101 "AMD RX 7800 XT" FAMILY dgpu-all gfx110X-all gfx110X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
)
therock_add_amdgpu_target(gfx1102 "AMD RX 7700S/Framework Laptop 16" FAMILY dgpu-all gfx110X-all gfx110X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
)
therock_add_amdgpu_target(gfx1103 "AMD Radeon 780M Laptop iGPU" FAMILY igpu-all gfx110X-all gfx110X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl  # https://github.com/ROCm/TheRock/issues/150
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)

# gfx115X family
therock_add_amdgpu_target(gfx1150 "AMD Strix Point iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl  # https://github.com/ROCm/TheRock/issues/150
)
therock_add_amdgpu_target(gfx1151 "AMD Strix Halo iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl  # https://github.com/ROCm/TheRock/issues/150
)
therock_add_amdgpu_target(gfx1152 "AMD Krackan 1 iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl  # https://github.com/ROCm/TheRock/issues/150
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)
therock_add_amdgpu_target(gfx1153 "AMD Radeon 820M iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
    rccl  # https://github.com/ROCm/TheRock/issues/150
    rocWMMA # https://github.com/ROCm/TheRock/issues/1944
)

# gfx120X family
therock_add_amdgpu_target(gfx1200 "AMD RX 9060 / XT" FAMILY dgpu-all gfx120X-all
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
)
therock_add_amdgpu_target(gfx1201 "AMD RX 9070 / XT" FAMILY dgpu-all gfx120X-all
  EXCLUDE_TARGET_PROJECTS
    hipSPARSELt # https://github.com/ROCm/TheRock/issues/2042
)

# Optional extension targets (used for out of tree target development).
include(therock_custom_amdgpu_targets OPTIONAL)

# Validates and normalizes AMDGPU target selection cache variables.
#
# This function handles two separate target lists:
#   THEROCK_AMDGPU_TARGETS: Per-architecture targets for architecture-specific builds
#   THEROCK_DIST_AMDGPU_TARGETS: All targets for the distribution (used by
#     runtime components that need to support all architectures)
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
  # If THEROCK_DIST_AMDGPU_FAMILIES is not set, it defaults to THEROCK_AMDGPU_FAMILIES.
  set(_dist_families "${THEROCK_DIST_AMDGPU_FAMILIES}")
  if(NOT _dist_families)
    set(_dist_families "${THEROCK_AMDGPU_FAMILIES}")
  endif()
  set(_dist_expanded_targets)
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
