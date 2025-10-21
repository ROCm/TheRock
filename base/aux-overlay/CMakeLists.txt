# Auxiliary Overlay
#
# This project contains some odds and ends needed in order to complete
# assembly of a full SDK. This includes:
#
# * Various redirect files to account for where things live in the tree from
# a user perspective.
# * Symlinks to account for distribution-level layout changes.
# * Other hacks as needed.

cmake_minimum_required(VERSION 3.25)

project(therock-aux-overlay)

if(NOT WIN32)
  add_custom_target(symlinks ALL
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "lib/llvm" "${CMAKE_CURRENT_BINARY_DIR}/llvm"
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/llvm"
    DESTINATION .
  )
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/overlay/" DESTINATION ".")

# Generate distribution info.
string(JSON dist_info_json_str SET "{}"
  "dist_amdgpu_targets" "\"${GPU_TARGETS}\""
)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dist_info.json" "${dist_info_json_str}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dist_info.json"
  DESTINATION "share/therock"
)

# ---------------------------------------------------------------------------
# TheRock manifest generation
# ---------------------------------------------------------------------------

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(THEROCK_MANIFEST_JSON "${CMAKE_CURRENT_BINARY_DIR}/therock_manifest.json")
set(THEROCK_MANIFEST_SCRIPT "${THEROCK_SOURCE_DIR}/build_tools/generate_therock_manifest.py")

add_custom_command(
  OUTPUT "${THEROCK_MANIFEST_JSON}"
  COMMAND "${Python3_EXECUTABLE}" "${THEROCK_MANIFEST_SCRIPT}" -o "${THEROCK_MANIFEST_JSON}"
  WORKING_DIRECTORY "${THEROCK_SOURCE_DIR}"
  DEPENDS "${THEROCK_MANIFEST_SCRIPT}" "${THEROCK_SOURCE_DIR}/.gitmodules"
  COMMENT "Generating TheRock manifest"
  VERBATIM
)

add_custom_target(aux-overlay-manifest ALL DEPENDS "${THEROCK_MANIFEST_JSON}")

install(FILES "${THEROCK_MANIFEST_JSON}" DESTINATION "share/therock")
