if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    # TODO: mirror full release and use therock_subproject_fetch, this is for TESTING ONLY

    if(EXISTS "${_source_dir}")
        file(REMOVE_RECURSE "${_source_dir}")
    endif()

    execute_process(
        COMMAND git clone --recursive --branch v5.0.9 https://github.com/open-mpi/ompi.git "${_source_dir}"
        RESULT_VARIABLE GIT_CLONE_RESULT
        OUTPUT_VARIABLE GIT_CLONE_OUT
        ERROR_VARIABLE GIT_CLONE_ERR
    )

    if(NOT GIT_CLONE_RESULT EQUAL 0)
      message(FATAL_ERROR "Git clone failed:\n${GIT_CLONE_LOG}\n${GIT_CLONE_ERR}\n")
    endif()

    execute_process(
        COMMAND git submodule update --init --recursive
        WORKING_DIRECTORY "${_source_dir}"
        RESULT_VARIABLE GIT_SUBMODULE_RESULT
        OUTPUT_VARIABLE GIT_SUBMODULE_OUT
        ERROR_VARIABLE GIT_SUBMODULE_ERR
    )

    if(NOT GIT_SUBMODULE_RESULT EQUAL 0)
        message(FATAL_ERROR "Git submodule update failed:\n${GIT_SUBMODULE_OUT}\n${GIT_SUBMODULE_ERR}")
    endif()

    # Touch the download stamp
    file(WRITE "${_download_stamp}" "")

    therock_cmake_subproject_declare(therock-openmpi
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
  COMMAND
    "${CMAKE_COMMAND}" -E chdir ${SOURCE_DIR} perl autogen.pl
  COMMAND
    "${SOURCE_DIR}/configure" "--prefix=${MPI_PREFIX}" --disable-silent-rules
  COMMAND
    make -j "${PAR_JOBS}"
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
