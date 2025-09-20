cmake_minimum_required(VERSION 3.25)
project(OpenBLAS_BUILD)

# Note that we install to the top-level but want our development assets and Posix
# runfiles to install to lib/host-math. Since we are building shared, on Windows,
# we want the DLL to go to the root bin/ dir so that it can be dynamically
# loaded by ROCm binaries. This is a bit of an abuse.
if(WIN32)
  set(CMAKE_INSTALL_BINDIR "bin")
else()
  set(CMAKE_INSTALL_BINDIR "lib/host-math/bin")
endif()
set(CMAKE_INSTALL_LIBDIR "lib/host-math/lib")
set(CMAKE_INSTALL_INCLUDEDIR "lib/host-math/include")

add_subdirectory(${SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/build-openblas)

# Alterations.
# We would like this to be a library scoped to ROCm, so give it a custom
# name.
# TODO: Also set Linux symbol versioning flags.
set_target_properties(openblas_shared PROPERTIES
  # Private SONAME/DLL name.
  OUTPUT_NAME rocm-openblas
  # Install location in the library is hard-coded to include in the install root.
  INTERFACE_INCLUDE_DIRECTORIES $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/openblas>
)

# Some of the test targets do not build with an altered library name on
# Windows. Just uniformly exclude them from the build.
set_target_properties(openblas_utest PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(openblas_utest_ext PROPERTIES EXCLUDE_FROM_ALL TRUE)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cblas-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cblas-config.cmake
  @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cblas-config.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenBLAS)
