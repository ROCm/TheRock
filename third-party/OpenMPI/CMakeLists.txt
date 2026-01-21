if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-openmpi-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.9.tar.bz2"
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/openmpi-5.0.9.tar.bz2"
      URL_HASH "SHA256=dfb72762531170847af3e4a0f21d77d7b23cf36f67ce7ce9033659273677d80b"
      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-openmpi
        FPRINT_SOURCE_DIR "${_source_dir}"
        FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
        USE_DIST_AMDGPU_TARGETS
        BACKGROUND_BUILD
        NO_MERGE_COMPILE_COMMANDS
        EXCLUDE_FROM_ALL
        OUTPUT_ON_FAILURE
        EXTERNAL_SOURCE_DIR
          .
        CMAKE_ARGS
        "-DSOURCE_DIR=${_source_dir}"
        BINARY_DIR
          build
        INSTALL_DESTINATION
          lib/MPI
        EXTRA_DEPENDS
          "${_download_stamp}"
    )

    therock_cmake_subproject_provide_package(therock-openmpi MPI lib/MPI/lib/cmake/OpenMPI)

    therock_cmake_subproject_activate(therock-openmpi)
    return()
endif()

include(CMakePackageConfigHelpers)

cmake_minimum_required(VERSION 3.25)
project(OpenMPI_BUILD VERSION 5.0.9)

set(MPI_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install")
set(MPI_INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}")

include(ProcessorCount)
ProcessorCount(PAR_JOBS)
if(NOT PAR_JOBS OR PAR_JOBS EQUAL 0)
  set(PAR_JOBS 1)
endif()

add_custom_target(
  build ALL
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  # release 5.0.9 uses autoconf 1.16.5, while the manylinux image uses 1.16.1, so we regenerate the autotools files due to incompatibility issues
  COMMAND
    ${CMAKE_COMMAND} -E chdir ${SOURCE_DIR} perl autogen.pl --force
  COMMAND
    ${SOURCE_DIR}/configure --prefix=${MPI_PREFIX} --disable-silent-rules
  COMMAND
    make -j ${PAR_JOBS}
  COMMAND
    make install
  VERBATIM
)

install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/install/"
  DESTINATION lib
  PATTERN "include" EXCLUDE
)

install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/install/include/"
  DESTINATION include
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mpi-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mpi-config.cmake"
    INSTALL_DESTINATION lib/cmake/OpenMPI
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mpi-version-config.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mpi-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mpi-version-config.cmake"
    DESTINATION lib/cmake/OpenMPI
)
