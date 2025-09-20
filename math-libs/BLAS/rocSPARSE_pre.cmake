if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).
  # rocSPARSE has deprecated dependencies on roctracer. We apply a patch to redirect
  # naked linking against `-lroctx64` to an explicitly found version of the library.
  # See: https://github.com/ROCm/TheRock/issues/364
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)
  find_library(_therock_legacy_roctx64 roctx64 REQUIRED)
  cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "${_therock_legacy_roctx64}")
endif()

# See the artifact descriptor where we require these matrices for the test
# artifact. Consider installing as part of the main project.
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/clients/matrices"
  DESTINATION "clients"
)
