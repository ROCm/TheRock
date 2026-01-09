# Add the plugin engines directory to the private install RPATH dirs for the unit tests that use the plugin.so
list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib/hipdnn_plugins/engines")
