# Add the plugin engines directory to the private install RPATH dirs
# This will be picked up by therock_global_post_subproject.cmake and converted
# to origin-relative paths for all executable and shared library targets

list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib/hipdnn_plugins/engines")

message(STATUS "Added lib/hipdnn_plugins/engines to THEROCK_PRIVATE_INSTALL_RPATH_DIRS for MIOpen-plugin")
