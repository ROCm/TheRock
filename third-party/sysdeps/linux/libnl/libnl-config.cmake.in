# Traverse from lib/cmake/FOO -> the directory holding lib
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

# Core libnl library
if(NOT TARGET libnl::libnl)
  add_library(libnl::libnl SHARED IMPORTED)
  set_target_properties(libnl::libnl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libnl-3.so"
  )
endif()

# Generic netlink library (depends on core)
if(NOT TARGET libnl::genl)
  add_library(libnl::genl SHARED IMPORTED)
  set_target_properties(libnl::genl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libnl-genl-3.so"
    INTERFACE_LINK_LIBRARIES "libnl::libnl"
  )
endif()

# Route library (depends on core)
if(NOT TARGET libnl::route)
  add_library(libnl::route SHARED IMPORTED)
  set_target_properties(libnl::route PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libnl-route-3.so"
    INTERFACE_LINK_LIBRARIES "libnl::libnl"
  )
endif()

# Inet diag library (depends on core)
if(NOT TARGET libnl::idiag)
  add_library(libnl::idiag SHARED IMPORTED)
  set_target_properties(libnl::idiag PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libnl-idiag-3.so"
    INTERFACE_LINK_LIBRARIES "libnl::libnl"
  )
endif()

# Netfilter library (depends on core and route)
if(NOT TARGET libnl::nf)
  add_library(libnl::nf SHARED IMPORTED)
  set_target_properties(libnl::nf PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libnl-nf-3.so"
    INTERFACE_LINK_LIBRARIES "libnl::libnl;libnl::route"
  )
endif()

# XFRM library (depends on core)
if(NOT TARGET libnl::xfrm)
  add_library(libnl::xfrm SHARED IMPORTED)
  set_target_properties(libnl::xfrm PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libnl-xfrm-3.so"
    INTERFACE_LINK_LIBRARIES "libnl::libnl"
  )
endif()

set(_IMPORT_PREFIX)
