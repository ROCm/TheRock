# Add the plugin engines directory to the private install RPATH dirs for the unit tests that use the plugin.so
list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib/hipdnn_plugins/engines")

# The plugin library is installed in lib/hipdnn_plugins/engines/, and we need to set origin properly for the RPATH to work
set_target_properties(fusilli_plugin PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")

# Verify IREE source is at an exact git tag. Fusilli currently uses pip
# installation of `iree-base-compiler` for its `iree-compile` executable
# dependency (run in subprocess). The pip package is only produced on git tags.
# Given the current dependencies it's convenient to restrict IREE dependency to
# tags, and align the tag used between IREE source and the pip package. Fusilli
# will eventually use `libIREECompiler.so` built in TheRock, after which there
# will be no reason to keep to tags.
execute_process(
  COMMAND git describe --tags --exact-match
  WORKING_DIRECTORY ${IREE_SOURCE_DIR}
  OUTPUT_VARIABLE _iree_git_tag
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _iree_tag_result
  ERROR_QUIET
)

if(_iree_tag_result)
  message(FATAL_ERROR
    "IREE source at ${IREE_SOURCE_DIR} is not at an exact git tag. "
    "Fusilli plugin tests require a tagged IREE release to install matching iree-base-compiler."
  )
endif()

# Strip "iree-" prefix to get pip-compatible version
string(REGEX REPLACE "^iree-" "" _iree_pip_version "${_iree_git_tag}")

# Write version file to build directory (configure time)
set(_iree_tag_file "${CMAKE_CURRENT_BINARY_DIR}/iree_tag_for_pip.txt")
file(WRITE "${_iree_tag_file}" "${_iree_pip_version}\n")

# Install alongside CTestTestfile.cmake
install(FILES "${_iree_tag_file}"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/fusilli_plugin_test_infra")
