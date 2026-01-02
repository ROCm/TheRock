# This finder resolves the virtual BLAS package for sub-projects.
# It defers to the built host-blas, if available, otherwise, failing.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("OpenBLAS" IN_LIST THEROCK_PROVIDED_PACKAGES)
  cmake_policy(POP)
  message(STATUS "Resolving bundled host-blas library from super-project")

  if(DEFINED BLA_SIZEOF_INTEGER AND BLA_SIZEOF_INTEGER EQUAL 8)
    find_package(OpenBLAS64 CONFIG REQUIRED)
    set(BLAS_LIBRARIES OpenBLAS64::OpenBLAS)
    add_library(BLAS::BLAS ALIAS OpenBLAS64::OpenBLAS)
  else()
    find_package(OpenBLAS CONFIG REQUIRED)
    set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
    add_library(BLAS::BLAS ALIAS OpenBLAS::OpenBLAS)
  endif()

  # See: https://cmake.org/cmake/help/latest/module/FindBLAS.html
  set(BLAS_LINKER_FLAGS)
  set(BLAS95_LIBRARIES)
  set(BLAS95_FOUND FALSE)
  set(BLAS_FOUND TRUE)
else()
  cmake_policy(POP)
  set(BLAS_FOUND FALSE)
endif()
