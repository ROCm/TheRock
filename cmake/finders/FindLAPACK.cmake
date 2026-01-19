# This finder resolves the virtual BLAS package for sub-projects.
# It defers to the built host-blas, if available, otherwise, failing.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("OpenBLAS" IN_LIST THEROCK_PROVIDED_PACKAGES)
  cmake_policy(POP)
  message(STATUS "Resolving bundled host-blas library from super-project")

  if(DEFINED BLA_SIZEOF_INTEGER AND BLA_SIZEOF_INTEGER EQUAL 8)
    find_package(OpenBLAS64 CONFIG REQUIRED)
    set(_OPENBLAS OpenBLAS64)
  else()
    find_package(OpenBLAS CONFIG REQUIRED)
    set(_OPENBLAS OpenBLAS)
  endif()
  
  set(LAPACK_LIBRARIES ${_OPENBLAS}::OpenBLAS)
  add_library(LAPACK::LAPACK ALIAS ${_OPENBLAS}::OpenBLAS)

  # See: https://cmake.org/cmake/help/latest/module/FindBLAS.html
  set(LAPACK_LINKER_FLAGS)
  set(LAPACK95_LIBRARIES)
  set(LAPACK95_FOUND FALSE)
  set(LAPACK_FOUND TRUE)
else()
  cmake_policy(POP)
  set(LAPACK_FOUND FALSE)
endif()
