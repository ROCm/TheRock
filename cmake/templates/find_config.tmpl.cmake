# Template find_package config file used to trampoline to a fixed location.
# Processed via configure_file with the following variables in scope:
#   @_package_name@ : Name of the package being searched
#   @_package_name_lower@ : Lowercase name of the package being searched
#   @_find_package_path@ : Absolute path to the directory we should redirect to

# This trampolines through the list of file patterns for config scripts that
# is prescribed in the find_package docs. Since this is presumed to be for
# a component of the super project that must exist, it is a fatal error if a
# suitable destination is not found.

if(EXISTS "@_find_package_path@/@_package_name@Config.cmake")
  message(STATUS "Super-project find_package(${CMAKE_FIND_PACKAGE_NAME}) -> @_find_package_path@/@_package_name@Config.cmake")
  include("@_find_package_path@/@_package_name@Config.cmake")
elseif(EXISTS "@_find_package_path@/@_package_name_lower@-config.cmake")
  message(STATUS "Super-project find_package(${CMAKE_FIND_PACKAGE_NAME}) -> @_find_package_path@/@_package_name_lower@-config.cmake")
  include("@_find_package_path@/@_package_name_lower@-config.cmake")
else()
  message(FATAL_ERROR "Super-project based find_package(@_package_name@) config "
    "file not found under @_find_package_path@"
  )
endif()
