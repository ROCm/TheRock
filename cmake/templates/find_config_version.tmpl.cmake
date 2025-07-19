# Template find_package version file used to trampoline to a fixed location.
# Processed via configure_file with the following variables in scope:
#   @_package_name@ : Name of the package being searched
#   @_package_name_lower@ : Lowercase name of the package being searched
#   @_find_package_path@ : Absolute path to the directory we should redirect to

# This trampolines through the list of file patterns for version scripts that
# is prescribed in the find_package docs.

if(EXISTS "@_find_package_path@/@_package_name@ConfigVersion.cmake")
  message(STATUS "Super-project find_package(${PACKAGE_FIND_NAME} VERSION ${PACKAGE_FIND_VERSION}) -> "
          "@_find_package_path@/@_package_name@ConfigVersion.cmake")
  include("@_find_package_path@/@_package_name@ConfigVersion.cmake")
elseif(EXISTS "@_find_package_path@/@_package_name@Config-version.cmake")
  message(STATUS "Super-project find_package(${PACKAGE_FIND_NAME} VERSION ${PACKAGE_FIND_VERSION}) -> "
          "@_find_package_path@/@_package_name@Config-version.cmake")
  include("@_find_package_path@/@_package_name@Config-version.cmake")
elseif(EXISTS "@_find_package_path@/@_package_name_lower@-configVersion.cmake")
  message(STATUS "Super-project find_package(${PACKAGE_FIND_NAME} VERSION ${PACKAGE_FIND_VERSION}) -> "
          "@_find_package_path@/@_package_name_lower@-configVersion.cmake")
  include("@_find_package_path@/@_package_name_lower@-configVersion.cmake")
elseif(EXISTS "@_find_package_path@/@_package_name_lower@-config-version.cmake")
  message(STATUS "Super-project find_package(${PACKAGE_FIND_NAME} VERSION ${PACKAGE_FIND_VERSION}) -> "
          "@_find_package_path@/@_package_name_lower@-config-version.cmake")
  include("@_find_package_path@/@_package_name_lower@-config-version.cmake")
else()
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()
