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

  # Fail-fast: Check if artifact is defined in topology
  if(DEFINED THEROCK_TOPOLOGY_ARTIFACTS)
    if(NOT "${slice_name}" IN_LIST THEROCK_TOPOLOGY_ARTIFACTS)
      message(FATAL_ERROR
        "Artifact '${slice_name}' is not defined in BUILD_TOPOLOGY.toml. "
        "All artifacts must be declared in the topology. "
        "Valid artifacts are: ${THEROCK_TOPOLOGY_ARTIFACTS}"
      )
    endif()
  endif()

  # Determine if this artifact should be split into generic + arch-specific components
  set(_should_split FALSE)
  if(THEROCK_KPACK_SPLIT_ARTIFACTS)
    set(_artifact_type "${THEROCK_ARTIFACT_TYPE_${slice_name}}")
    if(NOT _artifact_type)
      message(FATAL_ERROR
        "THEROCK_KPACK_SPLIT_ARTIFACTS is enabled but THEROCK_ARTIFACT_TYPE_${slice_name} "
        "is not defined. Ensure topology_to_cmake.py has been run."
      )
    endif()
    if("${_artifact_type}" STREQUAL "target-specific")
      set(_should_split TRUE)
      set(_split_databases "${THEROCK_ARTIFACT_SPLIT_DATABASES_${slice_name}}")
    endif()
  endif()

  # Normalize arguments.
  set(_target_name "artifact-${slice_name}")
  set(_archive_target_name "archive-${slice_name}")

  # Check if targets exist from topology (expected) vs duplicate definition (error)
  set(_target_exists FALSE)
  if(TARGET "${_target_name}")
    # Target exists - check if it's from topology or a duplicate
    # If THEROCK_TOPOLOGY_ARTIFACTS is defined, we expect the target to exist
    if(DEFINED THEROCK_TOPOLOGY_ARTIFACTS)
      set(_target_exists TRUE)
    else()
      message(FATAL_ERROR "Artifact slice '${slice_name}' provided more than once")
    endif()
  endif()
  if(TARGET "${_archive_target_name}")
    message(FATAL_ERROR "Archive slice '${slice_name}' provided more than once")
  endif()

  if(NOT ARG_DESCRIPTOR)
    set(ARG_DESCRIPTOR "artifact.toml")
  endif()
  cmake_path(ABSOLUTE_PATH ARG_DESCRIPTOR BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  file(SHA256 "${ARG_DESCRIPTOR}" _descriptor_fprint)

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

  # Compute fingerprint of dependencies.
  # TODO: Potentially prime content with some environment/machine state.
  set(_fprint_content "ARTIFACT=${slice_name}" "DESCRIPTOR=${_descriptor_fprint}")
  set(_fprint_is_valid TRUE)
  foreach(_subproject_dep ${ARG_SUBPROJECT_DEPS})
    get_target_property(_subproject_fprint "${_subproject_dep}" THEROCK_FPRINT)
    if(_subproject_fprint)
      list(APPEND _fprint_content "${_subproject_dep}=${_subproject_fprint}")
    else()
      message(STATUS "Cannot compute fprint for artifact ${slice_name} (no fprint for ${_subproject_dep})")
      set(_fprint_is_valid FALSE)
    endif()
  endforeach()
  set(_fprint)
  if(_fprint_is_valid)
    string(SHA256 _fprint "${_fprint_content}")
  endif()

  # Populate commands.
  set(_fileset_tool "${THEROCK_SOURCE_DIR}/build_tools/fileset_tool.py")
  set(_artifact_command
    COMMAND "${Python3_EXECUTABLE}" "${_fileset_tool}" artifact
          --root-dir "${THEROCK_BINARY_DIR}" --descriptor "${ARG_DESCRIPTOR}"
  )
  set(_flatten_command_list)
  set(_manifest_files)
  set(_component_dirs)

  # When splitting, populate to artifacts-unsplit/ first, then split to artifacts/
  if(_should_split)
    set(_artifacts_base_dir "${THEROCK_BINARY_DIR}/artifacts-unsplit")
  else()
    set(_artifacts_base_dir "${THEROCK_BINARY_DIR}/artifacts")
  endif()

  foreach(_component ${ARG_COMPONENTS})
    set(_component_dir "${_artifacts_base_dir}/${slice_name}_${_component}${_bundle_suffix}")
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

  # When splitting is enabled, run split_artifacts.py on each component
  if(_should_split)
    set(_split_tool "${THEROCK_KPACK_DIR}/python/rocm_kpack/tools/split_artifacts.py")
    set(_bundler_path "${THEROCK_BINARY_DIR}/compiler/amd-llvm/dist/lib/llvm/bin/clang-offload-bundler")
    set(_split_manifest_files)
    set(_split_component_dirs)

    foreach(_component ${ARG_COMPONENTS})
      set(_unsplit_component_dir "${_artifacts_base_dir}/${slice_name}_${_component}${_bundle_suffix}")
      set(_unsplit_manifest "${_unsplit_component_dir}/artifact_manifest.txt")
      set(_artifact_prefix "${slice_name}_${_component}")

      # The split output generic manifest (used as dependency tracking output)
      set(_split_generic_dir "${THEROCK_BINARY_DIR}/artifacts/${_artifact_prefix}_generic")
      set(_split_manifest "${_split_generic_dir}/artifact_manifest.txt")
      list(APPEND _split_manifest_files "${_split_manifest}")
      list(APPEND _split_component_dirs "${_split_generic_dir}")

      # Build split command arguments
      set(_split_command_args
        --artifact-dir "${_unsplit_component_dir}"
        --output-dir "${THEROCK_BINARY_DIR}/artifacts/"
        --artifact-prefix "${_artifact_prefix}"
        --clang-offload-bundler "${_bundler_path}"
      )
      if(_split_databases)
        list(APPEND _split_command_args --split-databases ${_split_databases})
      endif()

      add_custom_command(
        OUTPUT "${_split_manifest}"
        COMMENT "Splitting ${_artifact_prefix} into generic and arch-specific artifacts"
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${THEROCK_KPACK_DIR}/python"
          "${Python3_EXECUTABLE}" "${_split_tool}" ${_split_command_args}
        DEPENDS
          "${_unsplit_manifest}"
          "${_split_tool}"
      )
    endforeach()

    # IMPORTANT: Redirect downstream targets to depend on split artifacts.
    #
    # The split command depends on the unsplit manifest, preserving the
    # dependency chain: stage.stamp -> unsplit manifest -> split manifest.
    #
    # The splitter produces multiple outputs (generic + per-arch artifacts)
    # but only the generic manifest is tracked here. This means:
    # - Ninja will rebuild split artifacts when source changes
    # - If someone deletes an arch-specific artifact, ninja won't notice
    #   (the generic manifest still exists)
    # - This is acceptable since arch-specific artifacts are derived outputs
    set(_manifest_files ${_split_manifest_files})
    set(_component_dirs ${_split_component_dirs})
  endif()

  # If target exists from topology, create a helper target for file dependencies
  if(_target_exists)
    # Target already exists from topology - create a helper target for file dependencies
    add_custom_target(
      "${_target_name}_files"
      DEPENDS ${_manifest_files}
    )
    add_dependencies("${_target_name}" "${_target_name}_files")
  else()
    # Create new target (fallback for when topology is not loaded)
    add_custom_target(
      "${_target_name}"
      DEPENDS ${_manifest_files}
    )
  endif()
  add_dependencies(therock-artifacts "${_target_name}")
  if(ARG_DISTRIBUTION)
    add_dependencies("dist-${ARG_DISTRIBUTION}" "${_target_name}")
  endif()

  # Generate artifact archive commands and save fingerprints.
  #
  # NOTE: In the multi-arch CI flow that kpack splitting enables, archive
  # generation moves out of the build system and into the upload phase.
  # Once fully transitioned to that model, archive generation logic here
  # can be removed entirely.
  #
  # For now, skip archive generation when splitting is enabled since split
  # produces generic + per-arch directories with different naming conventions
  # that the current archive loop doesn't handle.
  set(_archive_files)
  set(_archive_sha_files)
  set(_artifacts_dir "${THEROCK_BINARY_DIR}/artifacts")
  file(MAKE_DIRECTORY "${_artifacts_dir}")
  if(_should_split)
    message(STATUS "Skipping archive generation for split artifact: ${slice_name}")
  endif()
  foreach(_component ${ARG_COMPONENTS})
    if(_should_split)
      continue()
    endif()
    set(_component_dir "${_artifacts_dir}/${slice_name}_${_component}${_bundle_suffix}")
    set(_fprint_file "${_component_dir}.fprint")
    if(_fprint_is_valid)
      file(WRITE "${_fprint_file}" "${_fprint}")
    elseif(EXISTS "${_fprint_file}")
      file(REMOVE "${_fprint_file}")
    endif()
    set(_manifest_file "${_component_dir}/artifact_manifest.txt")
    set(_archive_file "${_component_dir}${THEROCK_ARTIFACT_ARCHIVE_SUFFIX}.tar.xz")
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
