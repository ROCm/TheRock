# CMAKE_CURRENT_LIST_FILE is installed at:
#   <prefix>/lib/rocm_sysdeps/lib/cmake/libiberty/libiberty-config.cmake
# Strip the known trailing suffix to arrive at <prefix>/lib/rocm_sysdeps/.
string(REGEX REPLACE "/lib/cmake/libiberty/?$" "" _IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}")
if(_IMPORT_PREFIX STREQUAL "/")
  set(_IMPORT_PREFIX "")
endif()

set(LibIberty_FOUND TRUE)
set(LibIberty_LIBRARIES "${_IMPORT_PREFIX}/lib/libiberty.so")
set(LibIberty_INCLUDE_DIRS "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include/libiberty")
set(LibIberty_LIBRARY_DIRS "${_IMPORT_PREFIX}/lib")

if(NOT TARGET LibIberty::LibIberty)
  add_library(LibIberty::LibIberty SHARED IMPORTED GLOBAL)
  set_target_properties(LibIberty::LibIberty PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include;${_IMPORT_PREFIX}/include/libiberty"
    IMPORTED_LOCATION "${_IMPORT_PREFIX}/lib/libiberty.so"
  )
endif()

set(_IMPORT_PREFIX)
