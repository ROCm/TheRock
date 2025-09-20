cmake_minimum_required(VERSION 3.25)
project(ZSTD_BUILD)
include(ProcessorCount)
ProcessorCount(PAR_JOBS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # On Linux we do a shared library build with post-install patching for symbol
  # versioning and SONAME privatizing.
  set(CONFIGURE_ARGS
    -DZSTD_BUILD_STATIC=OFF
    -DZSTD_BUILD_SHARED=ON
    "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
  )
  set(POST_INSTALL_COMMANDS
    COMMAND
      "${CMAKE_COMMAND}" -E env
        "PATCHELF=${PATCHELF}"
        "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
        "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
      bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # On Windows, we build static.
  set(CONFIGURE_ARGS
    -DZSTD_BUILD_STATIC=ON
    -DZSTD_BUILD_SHARED=OFF
  )
  set(POST_INSTALL_COMMANDS)
else()
  message(FATAL_ERROR "System not recognized")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    "${CMAKE_COMMAND}"
      "-G${CMAKE_GENERATOR}"
      "-S${SOURCE_DIR}/build/cmake"
      "-B${CMAKE_CURRENT_BINARY_DIR}/b"
      "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${CMAKE_CURRENT_SOURCE_DIR}/zstd_include.cmake"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
      -DZSTD_BUILD_TESTS=OFF
      -DZSTD_LEGACY_SUPPORT=OFF
      -DZSTD_BUILD_PROGRAMS=OFF
      ${CONFIGURE_ARGS}
  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/b"
  COMMAND
    "${CMAKE_COMMAND}" --install "${CMAKE_CURRENT_BINARY_DIR}/b"
  ${POST_INSTALL_COMMANDS}
)
