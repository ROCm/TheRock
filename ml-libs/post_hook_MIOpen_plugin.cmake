# Add the plugin engines directory to the private install RPATH dirs for the unit tests that use the plugin.so
list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib/hipdnn_plugins/engines")

# The plugin library is installed in lib/hipdnn_plugins/engines/, and we need to set origin properly for the RPATH to work
set_target_properties(miopen_legacy_plugin PROPERTIES
    THEROCK_INSTALL_RPATH_ORIGIN "lib/hipdnn_plugins/engines")
