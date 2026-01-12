if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # When included in TheRock, we download sources and set up the sub-project.
  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  # TODO: Upload this to our internal S3
  therock_subproject_fetch(therock-libatomic-sources
    SOURCE_DIR "${_source_dir}"
    # Originally from: https://ftp.gnu.org/gnu/gcc/gcc-13.2.0/gcc-13.2.0.tar.gz
    URL "https://ftp.gnu.org/gnu/gcc/gcc-13.2.0/gcc-13.2.0.tar.gz"
    URL_HASH "SHA256=8cb4be3796651976f94b9356fa08d833524f62420d6292c5033a9a26af315078"
    TOUCH "${_download_stamp}"
  )

  therock_cmake_subproject_declare(therock-libatomic
    USE_DIST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR .
    FPRINT_SOURCE_DIR "${_source_dir}"
    FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
    BINARY_DIR build
    NO_MERGE_COMPILE_COMMANDS
    BACKGROUND_BUILD
    OUTPUT_ON_FAILURE
    CMAKE_ARGS
      "-DSOURCE_DIR=${_source_dir}"
      "-DPATCHELF=${PATCHELF}"
      "-DTHEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
    INSTALL_DESTINATION
      lib/rocm_sysdeps
    INTERFACE_INCLUDE_DIRS
      lib/rocm_sysdeps/include
    INTERFACE_LINK_DIRS
      lib/rocm_sysdeps/lib
    INTERFACE_INSTALL_RPATH_DIRS
      lib/rocm_sysdeps/lib
    INTERFACE_PKG_CONFIG_DIRS
      lib/rocm_sysdeps/lib/pkgconfig
    EXTRA_DEPENDS
      "${_download_stamp}"
  )
  therock_cmake_subproject_provide_package(therock-libatomic Libatomic lib/rocm_sysdeps/lib/cmake/Libatomic)
  therock_cmake_subproject_activate(therock-libatomic)

  therock_test_validate_shared_lib(
    PATH build/dist/lib/rocm_sysdeps/lib
    LIB_NAMES libatomic.so
  )
  return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(LIBATOMIC_BUILD)

include(ProcessorCount)
ProcessorCount(PAR_JOBS)
if(NOT PAR_JOBS OR PAR_JOBS EQUAL 0)
  set(PAR_JOBS 1)
endif()

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}" "${CMAKE_CURRENT_BINARY_DIR}/build_libatomic"
  COMMAND
    # Create build directory for libatomic
    "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/build_libatomic"
  COMMAND
    # Patch source for symbol versioning
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh" "${SOURCE_DIR}"
  COMMAND
    # Configure libatomic only from GCC sources
    bash -c "cd ${CMAKE_CURRENT_BINARY_DIR}/build_libatomic && ${SOURCE_DIR}/libatomic/configure --prefix=${CMAKE_INSTALL_PREFIX} --enable-shared --disable-static --libdir=${CMAKE_INSTALL_PREFIX}/lib"
  COMMAND
    # Build libatomic
    make -C "${CMAKE_CURRENT_BINARY_DIR}/build_libatomic" -j "${PAR_JOBS}"
  COMMAND
    # Install libatomic
    make -C "${CMAKE_CURRENT_BINARY_DIR}/build_libatomic" install
  COMMAND
    # Patch installed libraries for symbol prefixing
    "${CMAKE_COMMAND}" -E env
      "PATCHELF=${PATCHELF}"
      "THEROCK_SOURCE_DIR=${THEROCK_SOURCE_DIR}"
      "Python3_EXECUTABLE=${Python3_EXECUTABLE}" --
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh" "${CMAKE_INSTALL_PREFIX}"

  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_source.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/patch_install.sh"
    "${CMAKE_CURRENT_SOURCE_DIR}/version.lds"
  VERBATIM
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libatomic-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/libatomic-config.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libatomic-config.cmake" DESTINATION lib/cmake/Libatomic)

