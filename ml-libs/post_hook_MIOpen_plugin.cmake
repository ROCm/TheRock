# Add the plugin engines directory to the private install RPATH dirs
# This will be picked up by therock_global_post_subproject.cmake and converted
# to origin-relative paths for all executable and shared library targets

list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib/hipdnn_plugins/engines")

message(STATUS "Added lib/hipdnn_plugins/engines to THEROCK_PRIVATE_INSTALL_RPATH_DIRS for MIOpen_plugin")

# The plugin library is installed in lib/hipdnn_plugins/engines/, which is 2 levels deep
# We need to tell the build system this so it can compute correct RPATH from $ORIGIN/../../lib


set_target_properties(miopen_legacy_plugin PROPERTIES 
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")
    
message(STATUS "Set THEROCK_INSTALL_RPATH_ORIGIN for miopen_legacy_plugin")
