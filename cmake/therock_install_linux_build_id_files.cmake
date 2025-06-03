# This script is included during cmake_install.cmake at the top level of any
# sub-project which is seeking to split its debug files into a .build-id
# root folder on Linux.
# It runs with the following variables in scope:
#   THEROCK_DEBUG_BUILD_ID_PATHS: Absolute paths to binaries that are expected
#     to contain a Build ID for separating out debug info.
#   THEROCK_OBJCOPY: `objcopy` command found at configure time.
#   THEROCK_READELF: `readelf` command found at configure time.
#   THEROCK_STAGE_INSTALL_ROOT: The root of the stage installation tree. This
#     may be different from the CMAKE_INSTALL_PREFIX for projects that install
#     into a sub-tree within the overall tree.
# For each binary, inspects it to find the Build ID from an ELF section like:
#
#   Displaying notes found in: .note.gnu.build-id
#   Owner                Data size        Description
#   GNU                  0x00000014       NT_GNU_BUILD_ID (unique build ID bitstring)
#     Build ID: dac12d72d961c9d08880d946cd26618f0107dc4b
# And then installs the debug sections to an appropriate file in .build-id/

block(SCOPE_FOR VARIABLES)
  foreach(binary_path ${THEROCK_DEBUG_BUILD_ID_PATHS})
    if(NOT EXISTS "${binary_path}")
      message("Skipping debug info for ${binary_path} (does not exist)")
      continue()
    endif()

    execute_process(
      OUTPUT_VARIABLE elf_output
      COMMAND "${THEROCK_READELF}" -n "${binary_path}"
      COMMAND_ERROR_IS_FATAL ANY
    )
    string(REGEX MATCH "Build ID: ([0-9a-zA-Z][0-9a-zA-Z])([0-9a-zA-Z]+)" build_id_match "${elf_output}")
    if(NOT build_id_match)
      message(WARNING "Binary ${binary_path} contains no Build ID (possibly not built with compatible flags)")
      continue()
    endif()

    set(_build_id_prefix "${CMAKE_MATCH_1}")
    set(_build_id_suffix "${CMAKE_MATCH_2}")
    set(_output_path ".build-id/${_build_id_prefix}/${_build_id_suffix}.debug")
    message(STATUS "Installing debug info from ${binary_path} to ${_output_path}")

    set(_output_path "${THEROCK_STAGE_INSTALL_ROOT}/${_output_path}")
    cmake_path(GET _output_path PARENT_PATH _parent_dir)
    file(MAKE_DIRECTORY "${_parent_dir}")
    execute_process(
      COMMAND
        "${THEROCK_OBJCOPY}" --only-keep-debug
        "${binary_path}" "${_output_path}"
      COMMAND_ERROR_IS_FATAL ANY
    )
  endforeach()
endblock()
