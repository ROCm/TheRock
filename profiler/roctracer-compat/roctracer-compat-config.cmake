find_package(rocprofiler-sdk-roctx REQUIRED)

# Traverse from lib/cmake/FOO -> the directory holding lib
get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

# Trampoline library that exposes roctracer/roctx.h compatibility include
# and links to the rocprofiler-sdk backed real implementation.
if(NOT TARGET roctracer-compat::roctx)
  message(STATUS "Adding target roctracer-compat::roctx")
  add_library(roctracer-compat::roctx INTERFACE IMPORTED)
  set_target_properties(roctracer-compat::roctx PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
    INTERFACE_LINK_LIBRARIES rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library
  )
endif()

set(_IMPORT_PREFIX)
