if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-zstd-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/zstd-1.5.7.tar.gz"
      URL_HASH "SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-zstd
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-zstd zstd lib/rocm_sysdeps/lib/cmake/zstd)
    therock_cmake_subproject_activate(therock-zstd)

    therock_test_validate_shared_lib(
        PATH build/dist/lib/rocm_sysdeps/lib
        LIB_NAMES libzstd.so
    )

    return()
endif()

# Otherwise, this is the sub-project build.
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

# Apply patch source only on Linux
set(patch_source_commands)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND patch_source_commands   COMMAND
       bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    ${patch_source_commands}
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
