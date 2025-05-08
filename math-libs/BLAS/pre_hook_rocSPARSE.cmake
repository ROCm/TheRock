if(NOT WIN32)
  # Configure roctracer if on a supported operating system (Linux).

  # rocSPARSE has deprecated dependencies on roctracer. We apply a patch to move
  # these to rocprofiler-sdk via the compat library until it can be caught up.
  # See: https://github.com/ROCm/TheRock/issues/364
  find_package(roctracer-compat REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
  include(therock_subproject_utils)

  cmake_language(DEFER CALL therock_patch_linked_lib OLD_LIBRARY "roctx64" NEW_TARGET "roctracer-compat::roctx")
endif()
