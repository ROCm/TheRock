cmake_minimum_required(VERSION 3.25)
project(NUMACTL_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

find_program(ACLOCAL aclocal)
if(NOT ACLOCAL)
  message(FATAL_ERROR "numactl cannot be reliably built without autotools (on Ubuntu: apt install automake)")
endif()

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # We have to patch the sources so make a fresh copy.
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    # reconfigure the project as the version of autoconf available may differ
    autoreconf -f -i "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --disable-static
  COMMAND
    make -j "${PAR_JOBS}" V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/numa-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/numa-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/numa-config.cmake" DESTINATION lib/cmake/NUMA)
