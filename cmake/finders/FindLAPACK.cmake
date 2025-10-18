# This finder resolves the virtual BLAS package for sub-projects.
# It defers to the built host-blas, if available, otherwise, failing.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("OpenBLAS64" IN_LIST THEROCK_PROVIDED_PACKAGES)
  cmake_policy(POP)
  message(STATUS "Resolving bundled host-blas library from super-project")
  find_package(OpenBLAS64 CONFIG REQUIRED)
  # See: https://cmake.org/cmake/help/latest/module/FindBLAS.html
  set(LAPACK_LINKER_FLAGS)
  set(LAPACK_LIBRARIES OpenBLAS64::OpenBLAS64)
  add_library(LAPACK::LAPACK ALIAS OpenBLAS64::OpenBLAS64)
  set(LAPACK95_LIBRARIES)
  set(LAPACK95_FOUND FALSE)
  set(BLA_SIZEOF_INTEGER 8)
  set(BLA_VENDOR OpenBLAS)
  set(LAPACK_FOUND TRUE)
else()
  cmake_policy(POP)
  set(LAPACK_FOUND FALSE)
endif()
