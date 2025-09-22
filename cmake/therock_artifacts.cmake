# therock_artifacts.cmake
# Facilities for bundling artifacts for bootstrapping and subsequent CI/CD
# phases.

# therock_provide_artifact
# This populates directories under build/artifacts representing specific
# subsets of the install tree. See docs/development/artifacts.md for further
# design notes on the subsystem.
#
# While artifacts are the primary output of the build system, it is often
# an aid to development to materialize them all locally into a `distribution`.
# These are directories under build/dist/${ARG_DISTRIBUTION} (default "rocm").
#
# All artifact slices of a distribution should be non-overlapping, populating
# some subset of the install directory tree.
#
# This will produce the following convenience targets:
# - artifact-${slice_name} : Populate the build/artifacts/{qualified_name}
#   directory. Added as a dependency of the `therock-artifacts` target.
# - archive-${slice_name} : Populate the build/artifacts/{qualified_name}.tar.xz
#   archive file. Added as a dependency of the `therock-archives` target.
#
# Convenience targets with a "+expunge" suffix are created to remove corresponding
# files. Invoking the project level "expunge" will depend on all of them.
function(therock_provide_artifact slice_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    "TARGET_NEUTRAL"
    "DESCRIPTOR;DISTRIBUTION"
    "COMPONENTS;SUBPROJECT_DEPS"
  )

  if(NOT ${slice_name} MATCHES "^[A-Za-z][A-Za-z0-9-]*$")
    message(FATAL_ERROR
      "Artifact slice name '${slice_name}' must start with a letter "
      "and may only contain alphanumeric characters and dashes"
    )
  endif()

  # Normalize arguments.
  set(_target_name "artifact-${slice_name}")
  set(_archive_target_name "archive-${slice_name}")
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

  if(NOT DEFINED ARG_DISTRIBUTION)
    set(ARG_DISTRIBUTION "rocm")
  endif()

  if(ARG_DISTRIBUTION)
    set(_dist_dir "${THEROCK_BINARY_DIR}/dist/${ARG_DISTRIBUTION}")
    # First time we see the distribution, set up targets and install.
    if(NOT TARGET "dist-${ARG_DISTRIBUTION}")
      add_custom_target("dist-${ARG_DISTRIBUTION}" ALL)
      add_dependencies(therock-dist "dist-${ARG_DISTRIBUTION}")

      # expunge target for the dist
      add_custom_target(
        "dist-${ARG_DISTRIBUTION}+expunge"
        COMMAND
          "${CMAKE_COMMAND}" -E rm -rf "${_dist_dir}"
      )
      add_dependencies(therock-expunge "dist-${ARG_DISTRIBUTION}+expunge")

      # Add install()
      install(
        DIRECTORY "${_dist_dir}/"
        DESTINATION "."
        COMPONENT "${ARG_DISTRIBUTION}"
        USE_SOURCE_PERMISSIONS
      )

      add_custom_target(
        "install-${ARG_DISTRIBUTION}"
        COMMAND
          "${CMAKE_COMMAND}"
            --install "${THEROCK_BINARY_DIR}"
            --component "${ARG_DISTRIBUTION}"
        DEPENDS
          "dist-${ARG_DISTRIBUTION}"
      )
    endif()
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

  # Populate commands.
  set(_fileset_tool "${THEROCK_SOURCE_DIR}/build_tools/fileset_tool.py")
  set(_artifact_command
    COMMAND "${Python3_EXECUTABLE}" "${_fileset_tool}" artifact
          --root-dir "${THEROCK_BINARY_DIR}" --descriptor "${ARG_DESCRIPTOR}"
  )
  set(_flatten_command_list)
  set(_manifest_files)
  set(_component_dirs)
  foreach(_component ${ARG_COMPONENTS})
    set(_component_dir "${THEROCK_BINARY_DIR}/artifacts/${slice_name}_${_component}${_bundle_suffix}")
    list(APPEND _component_dirs "${_component_dir}")
    set(_manifest_file "${_component_dir}/artifact_manifest.txt")
    list(APPEND _manifest_files "${_manifest_file}")
    # The 'artifact' command takes an alternating list of component name and
    # directory to populate.
    list(APPEND _artifact_command
      "${_component}"
      "${_component_dir}"
    )
  endforeach()
  # Populate the corresponding build/dist/DISTRIBUTION directory.
  if(ARG_DISTRIBUTION)
    list(APPEND _flatten_command_list
      COMMAND "${Python3_EXECUTABLE}" "${_fileset_tool}" artifact-flatten
        -o "${_dist_dir}" ${_component_dirs}
    )
  endif()
  add_custom_command(
    OUTPUT ${_manifest_files}
    COMMENT "Populate artifact ${slice_name}"
    ${_artifact_command}
    ${_flatten_command_list}
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
  if(ARG_DISTRIBUTION)
    add_dependencies("dist-${ARG_DISTRIBUTION}" "${_target_name}")
  endif()

  # Generate artifact archive commands.
  set(_archive_files)
  set(_archive_sha_files)
  foreach(_component ${ARG_COMPONENTS})
    set(_component_dir "${THEROCK_BINARY_DIR}/artifacts/${slice_name}_${_component}${_bundle_suffix}")
    set(_manifest_file "${_component_dir}/artifact_manifest.txt")
    set(_archive_file "${THEROCK_BINARY_DIR}/artifacts/${slice_name}_${_component}${_bundle_suffix}${THEROCK_ARTIFACT_ARCHIVE_SUFFIX}.tar.xz")
    list(APPEND _archive_files "${_archive_file}")
    set(_archive_sha_file "${_archive_file}.sha256sum")
    list(APPEND _archive_sha_files "${_archive_sha_file}")
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

  # Archive expunge target.
  add_custom_target(
    "${_archive_target_name}+expunge"
    COMMAND
      "${CMAKE_COMMAND}" -E rm -f ${_archive_files} ${_archive_sha_files}
  )
  add_dependencies(therock-expunge "${_archive_target_name}+expunge")

  # Generate expunge targets.
  add_custom_target(
    "${_target_name}+expunge"
    COMMAND
      "${CMAKE_COMMAND}" -E rm -rf ${_component_dirs}
  )
  add_dependencies(therock-expunge "${_target_name}+expunge")
  add_dependencies("dist-${ARG_DISTRIBUTION}+expunge" "${_target_name}+expunge")

  # For each subproject dep, we add a dependency on its +dist target to also
  # trigger overall artifact construction. In this way `ninja myfoo+dist`
  # will always populate all related artifacts and distributions. Note that
  # this only applies to the convenience +dist target, not the underlying
  # stamp-file chain, which is what the core dependency mechanism uses.
  if(ARG_DISTRIBUTION)
    foreach(subproject_dep ${ARG_SUBPROJECT_DEPS})
      set(_subproject_dist_target "${subproject_dep}+dist")
      if(NOT TARGET "${_subproject_dist_target}")
        message(FATAL_ERROR "Subproject convenience target ${_subproject_dist_target} does not exist")
      endif()
      add_dependencies("${_subproject_dist_target}" "${_target_name}")
    endforeach()
  endif()
endfunction()
