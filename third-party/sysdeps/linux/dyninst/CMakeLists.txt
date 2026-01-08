if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # When included in TheRock, we download sources and set up the sub-project.
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-dyninst-sources
    SOURCE_DIR "${_source_dir}"
    # Originally mirrored from: https://github.com/dyninst/dyninst/archive/refs/tags/v13.0.0.tar.gz
    URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/dyninst-13.0.0.tar.gz
    URL_HASH "SHA512=49e98debe661a0c7be7c4ab2ad4e140d95ed82a01d17e672f2173d8c6763761797985e67c97f0663647745277f10110e538ea558b1518cd04789d77eb6cf4174"
    TOUCH "${_download_stamp}"
  )

  therock_cmake_subproject_declare(therock-dyninst
    BACKGROUND_BUILD
    EXTERNAL_SOURCE_DIR .
    FPRINT_SOURCE_DIR "${_source_dir}"
    FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
    BINARY_DIR build
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    CMAKE_ARGS
      "-DSOURCE_DIR=${_source_dir}"
    RUNTIME_DEPS
      therock-boost
      therock-tbb
      ${THEROCK_BUNDLED_BINUTILS}
      ${THEROCK_BUNDLED_ELFUTILS}
    INSTALL_RPATH_DIRS
      lib/rocm_sysdeps/lib
    EXTRA_DEPENDS
      "${_download_stamp}"
  )
  therock_cmake_subproject_provide_package(therock-dyninst Dyninst lib/rocm_sysdeps/lib/cmake)
  therock_cmake_subproject_activate(therock-dyninst)

  therock_test_validate_shared_lib(
    PATH build/dist/lib/rocm_sysdeps/lib
    LIB_NAMES libdyninstAPI_RT.so libdyninstAPI.so
  )

  add_dependencies(therock-third-party therock-dyninst)
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(DYNINST_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)

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
    "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_BINARY_DIR}/source" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}

  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh"
)
