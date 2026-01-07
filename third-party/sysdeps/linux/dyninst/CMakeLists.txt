if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # When included in TheRock, we download sources and set up the sub-project.
  therock_subproject_fetch(therock-dyninst-sources
    CMAKE_PROJECT
    # Originally mirrored from: https://github.com/dyninst/dyninst/archive/refs/tags/v13.0.0.tar.gz
    URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/dyninst-13.0.0.tar.gz
    URL_HASH SHA256=1bc48d26478b677a6c090c25586a447507bd1b4cf88d369bd61820005ce1be39
  )

  therock_cmake_subproject_declare(therock-dyninst
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
    RUNTIME_DEPS
      therock-boost
      therock-tbb
      ${THEROCK_BUNDLED_BINUTILS}
      ${THEROCK_BUNDLED_ELFUTILS}
    INSTALL_RPATH_DIRS
      lib/rocm_sysdeps/lib
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
    "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/s"
  COMMAND
    "${CMAKE_CURRENT_BINARY_DIR}/s/configure"
      --prefix "${CMAKE_INSTALL_PREFIX}"
      --libdir="${CMAKE_INSTALL_PREFIX}/lib"
      --includedir="${CMAKE_INSTALL_PREFIX}/include"
  COMMAND
    make -j ${PAR_JOBS} V=1
  COMMAND
    make install
  COMMAND
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" ${CMAKE_INSTALL_PREFIX}

  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh"
)
