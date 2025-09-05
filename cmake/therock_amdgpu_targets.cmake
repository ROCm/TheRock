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
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
)
therock_add_amdgpu_target(gfx908 "MI100 CDNA" FAMILY gfx90X-all dcgpu-all gfx90X-dcgpu)
therock_add_amdgpu_target(gfx90a "MI210/250 CDNA" FAMILY gfx90X-all dcgpu-all gfx90X-dcgpu)

# gfx94X family
therock_add_amdgpu_target(gfx942 "MI300A/MI300X CDNA" FAMILY dcgpu-all gfx94X-all gfx94X-dcgpu)

# gfx950
therock_add_amdgpu_target(gfx950 "MI350X/MI355X CDNA" FAMILY dcgpu-all gfx950-all gfx950-dcgpu)

# gfx101X family
therock_add_amdgpu_target(gfx1010 "AMD RX 5700" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
)
therock_add_amdgpu_target(gfx1011 "AMD Radeon Pro V520" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
)
therock_add_amdgpu_target(gfx1012 "AMD RX 5500" FAMILY dgpu-all gfx101X-all gfx101X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
    composable_kernel # https://github.com/ROCm/TheRock/issues/1245
)

# gfx103X family
therock_add_amdgpu_target(gfx1030 "AMD RX 6800 / XT" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
)
therock_add_amdgpu_target(gfx1032 "AMD RX 6600" FAMILY dgpu-all gfx103X-all gfx103X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
)
therock_add_amdgpu_target(gfx1035 "AMD Radeon 680M Laptop iGPU" igpu-all FAMILY gfx103X-all gfx103X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
)
therock_add_amdgpu_target(gfx1036 "AMD Raphael iGPU" FAMILY igpu-all gfx103X-all gfx103X-igpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
)

# gfx110X family
therock_add_amdgpu_target(gfx1100 "AMD RX 7900 XTX" FAMILY dgpu-all gfx110X-all gfx110X-dgpu)
therock_add_amdgpu_target(gfx1101 "AMD RX 7800 XT" FAMILY dgpu-all gfx110X-all gfx110X-dgpu)
therock_add_amdgpu_target(gfx1102 "AMD RX 7700S/Framework Laptop 16" FAMILY dgpu-all gfx110X-all gfx110X-dgpu
  EXCLUDE_TARGET_PROJECTS
    hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
)
therock_add_amdgpu_target(gfx1103 "AMD Radeon 780M Laptop iGPU" FAMILY igpu-all gfx110X-all gfx110X-igpu
  EXCLUDE_TARGET_PROJECTS
    rccl  # https://github.com/ROCm/TheRock/issues/150
)

# gfx115X family
therock_add_amdgpu_target(gfx1150 "AMD Strix Point iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    rccl  # https://github.com/ROCm/TheRock/issues/150
)
therock_add_amdgpu_target(gfx1151 "AMD Strix Halo iGPU" FAMILY igpu-all gfx115X-all gfx115X-igpu
  EXCLUDE_TARGET_PROJECTS
    rccl  # https://github.com/ROCm/TheRock/issues/150
)

# gfx120X family
therock_add_amdgpu_target(gfx1200 "AMD RX 9060 / XT" FAMILY dgpu-all gfx120X-all)
therock_add_amdgpu_target(gfx1201 "AMD RX 9070 / XT" FAMILY dgpu-all gfx120X-all)


# Validates and normalizes AMDGPU target selection cache variables.
function(therock_validate_amdgpu_targets)
  message(STATUS "Configured AMDGPU Targets:")
  string(APPEND CMAKE_MESSAGE_INDENT "  ")
  set(_expanded_targets)
  set(_explicit_selections)
  get_property(_available_families GLOBAL PROPERTY THEROCK_AMDGPU_TARGET_FAMILIES)
  list(REMOVE_DUPLICATES _available_families)
  get_property(_available_targets GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)
  # Expand families.
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

  # Validate targets.
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

  # Must have a target.
  if(NOT _expanded_targets)
    message(FATAL_ERROR
      "No AMDGPU target selected: make a selection via THEROCK_AMDGPU_FAMILIES "
      "or THEROCK_AMDGPU_TARGETS."
    )
  endif()
  # Export to parent scope.
  set(THEROCK_AMDGPU_TARGETS "${_expanded_targets}" PARENT_SCOPE)
  string(JOIN " " _expanded_targets_spaces ${_expanded_targets})
  set(THEROCK_AMDGPU_TARGETS_SPACES "${_expanded_targets_spaces}" PARENT_SCOPE)

  # Export the dist targets as the same until we have support/need to separate
  # them.
  set(THEROCK_DIST_AMDGPU_TARGETS "${_expanded_targets}" PARENT_SCOPE)
  set(THEROCK_DIST_AMDGPU_TARGETS_SPACES "${_expanded_targets_spaces}" PARENT_SCOPE)

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
    message(STATUS "* Dist bundle: ${_explicit_selections}")
  else()
    message(STATUS "* Dist bundle: ${THEROCK_AMDGPU_DIST_BUNDLE_NAME}")
  endif()
endfunction()

function(therock_get_amdgpu_target_name out_var gfx_target)
  get_property(_name GLOBAL PROPERTY "THEROCK_AMDGPU_TARGET_NAME_${gfx_target}")
  set("${out_var}" "${_name}" PARENT_SCOPE)
endfunction()
