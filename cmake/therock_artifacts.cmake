# therock_artifacts.cmake
# Facilities for bundling artifacts for bootstrapping and subsequent CI/CD
# phases.

# Property containing all artifact directories for all components created via
# therock_provide_artifact(). This is used to populate the dist/rocm directory
# by flattening them all. In the future, we may have multiple dist groups but
# there is just this one for now.
set_property(GLOBAL PROPERTY THEROCK_DIST_ARTIFACT_DIRS)

function(therock_provide_artifact slice_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    "TARGET_NEUTRAL"
    "DESCRIPTOR"
    "COMPONENTS;SUBPROJECT_DEPS"
  )

  # Normalize arguments.
  set(_target_name "therock-artifact-${slice_name}")
  set(_archive_target_name "therock-archive-${slice_name}")
  if(TARGET "${_target_name}")
    message(FATAL_ERROR "Artifact slice '${slice_name}' provided more than once")
  endif()
  if(TARGET "${_archive_target_name}")
    message(FATAL_ERROR "Archive slice '${slice_name}' provided more than once")
  endif()

  if(NOT ARG_DESCRIPTOR)
    set(ARG_DESCRIPTOR "artifact.toml")
  endif()
  cmake_path(ABSOLUTE_PATH ARG_DESCRIPTOR BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  # We make all artifact slices available as therock-artifacts top-level target.
  if(NOT TARGET therock-artifacts)
    add_custom_target(therock-artifacts)
  endif()
  if(NOT TARGET therock-archives)
    add_custom_target(therock-archives)
  endif()

  # Determine top-level name.
  if(ARG_TARGET_NEUTRAL)
    set(_bundle_suffix "_generic")
  else()
    set(_bundle_suffix "_${THEROCK_AMDGPU_DIST_BUNDLE_NAME}")
  endif()

  ### Generate artifact directories.
  # Determine dependencies.
  set(_stamp_file_deps)
  _therock_cmake_subproject_deps_to_stamp(_stamp_file_deps "stage.stamp" ${ARG_SUBPROJECT_DEPS})

  # Assemble commands.
  set(_fileset_tool "${THEROCK_SOURCE_DIR}/build_tools/fileset_tool.py")
  set(_command_list)
  set(_manifest_files)
  foreach(_component ${ARG_COMPONENTS})
    set(_component_dir "${THEROCK_BINARY_DIR}/artifacts/${slice_name}_${_component}${_bundle_suffix}")
    set_property(GLOBAL APPEND PROPERTY THEROCK_DIST_ARTIFACT_DIRS "${_component_dir}")
    set(_manifest_file "${_component_dir}/artifact_manifest.txt")
    list(APPEND _manifest_files "${_manifest_file}")
    list(APPEND _command_list
      COMMAND "${Python3_EXECUTABLE}" "${_fileset_tool}" artifact
        --output-dir "${_component_dir}"
        --root-dir "${THEROCK_BINARY_DIR}" --descriptor "${ARG_DESCRIPTOR}"
        --component "${_component}"
    )
  endforeach()

  # Set up command.
  add_custom_command(
    OUTPUT ${_manifest_files}
    COMMENT "Merging artifact ${slice_name}"
    ${_command_list}
    DEPENDS
      ${_stamp_file_deps}
      "${ARG_DESCRIPTOR}"
      "${_fileset_tool}"
  )
  add_custom_target(
    "${_target_name}"
    DEPENDS ${_manifest_files}
  )
  add_dependencies(therock-artifacts "${_target_name}")

  ### Generate artifact archive commands.
  set(_archive_files)
  foreach(_component ${ARG_COMPONENTS})
    set(_component_dir "${THEROCK_BINARY_DIR}/artifacts/${slice_name}_${_component}${_bundle_suffix}")
    set(_manifest_file "${_component_dir}/artifact_manifest.txt")
    set(_archive_file "${THEROCK_BINARY_DIR}/artifacts/${slice_name}_${_component}${_bundle_suffix}${THEROCK_ARTIFACT_ARCHIVE_SUFFIX}.tar.xz")
    list(APPEND _archive_files "${_archive_file}")
    set(_archive_sha_file "${_archive_file}.sha256sum")
    # TODO(#726): Lower compression levels are much faster for development and CI.
    #             Set back to 6+ for production builds?
    set(_archive_compression_level 2)
    add_custom_command(
      OUTPUT
        "${_archive_file}"
        "${_archive_sha_file}"
      COMMENT "Creating archive ${_archive_file}"
      COMMAND
        "${Python3_EXECUTABLE}" "${_fileset_tool}"
        artifact-archive "${_component_dir}"
          -o "${_archive_file}"
          --compression-level "${_archive_compression_level}"
          --hash-file "${_archive_sha_file}" --hash-algorithm sha256
      DEPENDS
        "${_manifest_file}"
        "${_fileset_tool}"
    )
  endforeach()

  add_custom_target("${_archive_target_name}" DEPENDS ${_archive_files})
  add_dependencies(therock-archives "${_archive_target_name}")
endfunction()


function(therock_create_dist)
  # Currently there is only one dist se we hard-code. These could become
  # settings later.
  set(_dist_dir "${THEROCK_BINARY_DIR}/dist/rocm")
  set(_stamp_file "${THEROCK_BINARY_DIR}/dist/.rocm.stamp")
  set(_dist_name "rocm")
  get_property(_artifact_dirs GLOBAL PROPERTY THEROCK_DIST_ARTIFACT_DIRS)

  set(_fileset_tool "${THEROCK_SOURCE_DIR}/build_tools/fileset_tool.py")
  list(TRANSFORM _artifact_dirs APPEND "/artifact_manifest.txt" OUTPUT_VARIABLE _manifest_files)

  add_custom_command(
    OUTPUT "${_stamp_file}"
    COMMENT "Creating dist ${_dist_dir}"
    COMMAND "${Python3_EXECUTABLE}" "${_fileset_tool}" artifact-flatten --verbose
      -o "${_dist_dir}" ${_artifact_dirs}
    COMMAND
      "${CMAKE_COMMAND}" -E touch "${_stamp_file}"
    DEPENDS
      "${_fileset_tool}"
      ${_manifest_files}
  )

  set(_dist_target_name "therock-dist-${_dist_name}")
  add_custom_target("${_dist_target_name}" ALL DEPENDS "${_stamp_file}")
  if(NOT TARGET therock-dist)
    add_custom_target(therock-dist ALL)
  endif()
  add_dependencies(therock-dist "${_dist_target_name}")
endfunction()
