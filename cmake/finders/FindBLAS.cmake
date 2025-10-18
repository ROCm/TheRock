# This finder resolves the virtual BLAS package for sub-projects.
# It defers to the built host-blas, if available, otherwise, failing.
cmake_policy(PUSH)
cmake_policy(SET CMP0057 NEW)

if("OpenBLAS64" IN_LIST THEROCK_PROVIDED_PACKAGES)
  cmake_policy(POP)
  message(STATUS "Resolving bundled host-blas library from super-project")
  find_package(OpenBLAS64 CONFIG REQUIRED)
  # See: https://cmake.org/cmake/help/latest/module/FindBLAS.html
  set(BLAS_LINKER_FLAGS)
  set(BLAS_LIBRARIES OpenBLAS64::OpenBLAS64)
  add_library(BLAS::BLAS ALIAS OpenBLAS64::OpenBLAS64)
  set(BLAS95_LIBRARIES)
  set(BLAS95_FOUND FALSE)
  set(BLA_SIZEOF_INTEGER 8)
  set(BLAS_FOUND TRUE)
else()
  cmake_policy(POP)
  set(BLAS_FOUND FALSE)
endif()
