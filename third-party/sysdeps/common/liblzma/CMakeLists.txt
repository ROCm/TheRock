if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    # Originally mirrored from: https://github.com/tukaani-project/xz/releases/download/v5.8.1/xz-5.8.1.tar.bz2
    set(_xz_url "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/xz-5.8.1.tar.bz2")
    therock_subproject_fetch(therock-liblzma-sources
      SOURCE_DIR "${_source_dir}"
      URL "${_xz_url}"
      URL_HASH SHA256=5965c692c4c8800cd4b33ce6d0f6ac9ac9d6ab227b17c512b6561bce4f08d47e
      TOUCH "${_download_stamp}"
    )

    # extract version from url
    string(REGEX MATCH "xz-([0-9]+\.[0-9]+\.[0-9]+)\.tar" _ "${_xz_url}")
    set(LIBLZMA_VERSION "${CMAKE_MATCH_1}")

    therock_cmake_subproject_declare(therock-liblzma
      EXTERNAL_SOURCE_DIR .
      BINARY_DIR build
      NO_MERGE_COMPILE_COMMANDS
      BACKGROUND_BUILD
      OUTPUT_ON_FAILURE
      CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        "-DPATCHELF=${PATCHELF}"
        "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
        "-DLIBLZMA_VERSION=${LIBLZMA_VERSION}"
      INSTALL_DESTINATION
        lib/rocm_sysdeps
      INTERFACE_PROGRAM_DIRS
        lib/rocm_sysdeps/bin
      INTERFACE_LINK_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_INSTALL_RPATH_DIRS
        lib/rocm_sysdeps/lib
      INTERFACE_PKG_CONFIG_DIRS
        lib/rocm_sysdeps/lib/pkgconfig
      EXTRA_DEPENDS
        "${_download_stamp}"
    )
    therock_cmake_subproject_provide_package(therock-liblzma liblzma lib/rocm_sysdeps/lib/cmake/liblzma)
    therock_cmake_subproject_activate(therock-liblzma)

    therock_test_validate_shared_lib(
      PATH build/dist/lib/rocm_sysdeps/lib
      LIB_NAMES liblzma.so
    )
    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBLZMA_BUILD VERSION "${LIBLZMA_VERSION}")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

# Apply patch source only on Linux
set(patch_source_commands)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND patch_source_commands   COMMAND
    bash "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${CMAKE_CURRENT_BINARY_DIR}/s")
endif()
# xz/liblzma provides a CMakeLists.txt, however, we have to do some post-processing
# of the libraries in order to prepare them for our use, so we invoke it as
# a sub-build. We do this uniformly across all platforms because it is easier
# to have common code paths. We write directly into the install prefix at
# build time since this build is only ever used in a one shot way.
add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    ${patch_source_commands}
  COMMAND
    # Provide our own version map with private symbol versions.
    "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/version.lds" "${CMAKE_CURRENT_BINARY_DIR}/s/src/liblzma/liblzma_generic.map"
  COMMAND
    "${CMAKE_COMMAND}"
      "-G${CMAKE_GENERATOR}"
      "-S${CMAKE_CURRENT_BINARY_DIR}/s"
      "-B${CMAKE_CURRENT_BINARY_DIR}/b"
      "-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${CMAKE_CURRENT_SOURCE_DIR}/liblzma_include.cmake"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
      # TODO: This should tunnel the CMAKE_BUILD_TYPE from TheRock
      "-DCMAKE_BUILD_TYPE=Release"
      # Disable symlink creation
      "-DCREATE_XZ_SYMLINKS=OFF"
      "-DCREATE_LZMA_SYMLINKS=OFF"
      # Disable writing pc files: we will do so ourselves.
      -DSKIP_INSTALL_FILES=ON
      # Set versioning to generic to override with a private map
      -DXZ_SYMBOL_VERSIONING=generic

  COMMAND
    "${CMAKE_COMMAND}" --build "${CMAKE_CURRENT_BINARY_DIR}/b"
  COMMAND
    "${CMAKE_COMMAND}" --install "${CMAKE_CURRENT_BINARY_DIR}/b"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}" --
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.py" ${CMAKE_INSTALL_PREFIX}
)


# Output cmake and pkgconfig files. Note that the library names are different
# on windows vs linux.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LIBLZMA_TYPE "SHARED")
  set(LIBLZMA_IMPORTED_LOCATION "lib/librocm_sysdeps_liblzma.so.5")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(LIBLZMA_TYPE "STATIC")
  set(LIBLZMA_IMPORTED_LOCATION "lib/liblzmastatic.lib")
else()
  message(FATAL_ERROR "Unrecognized system type")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/liblzma-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/liblzma-config.cmake
  @ONLY
)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/liblzma.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/liblzma.pc
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/liblzma-config.cmake" DESTINATION lib/cmake/LIBLZMA)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/liblzma.pc" DESTINATION lib/pkgconfig)
