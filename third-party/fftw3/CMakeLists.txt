# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # When included in TheRock, download sources and set up the sub-projects.
  # THEROCK_FLAG_FFTW3_AUTOTOOLS_BUILD selects an autotools build on Linux.
  # Otherwise the tarball is built with its own CMakeLists.txt.
  set(_fftw3_use_autotools OFF)
  if(THEROCK_FLAG_FFTW3_AUTOTOOLS_BUILD)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(_fftw3_use_autotools ON)
    else()
      message(WARNING
        "THEROCK_FLAG_FFTW3_AUTOTOOLS_BUILD is only supported on Linux; "
        "falling back to the upstream CMake build")
    endif()
  endif()

  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")

  if(_fftw3_use_autotools)
    set(_download_stamp "${_source_dir}/download.stamp")
    therock_subproject_fetch(therock-fftw3-sources
      SOURCE_DIR "${_source_dir}"
      # Originally mirrored from: https://fftw.org/fftw-3.3.10.tar.gz
      URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fftw-3.3.10.tar.gz
      URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
      TOUCH "${_download_stamp}"
      # Preserve extracted timestamps. Otherwise make treats the autotools
      # inputs as newer than configure and tries to regenerate them.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    set(_fftw3_external_source_dir .)
    set(_fftw3_common_cmake_args
      "-DSOURCE_DIR=${_source_dir}"
      -DENABLE_THREADS=ON
    )
    set(_fftw3_extra_declare_args
      FPRINT_SOURCE_DIR "${_source_dir}"
      FPRINT_FILE_GLOBS "${CMAKE_CURRENT_LIST_DIR}/*"
      EXTRA_DEPENDS "${_download_stamp}"
    )
  else()
    therock_subproject_fetch(therock-fftw3-sources
      CMAKE_PROJECT
      # Originally mirrored from: https://fftw.org/fftw-3.3.10.tar.gz
      URL https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/fftw-3.3.10.tar.gz
      URL_HASH SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467
    )

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(_fftw3_enable_threads "ON")
    else()
      set(_fftw3_enable_threads "OFF")
    endif()

    set(_fftw3_external_source_dir "${_source_dir}")
    set(_fftw3_common_cmake_args
      -DBUILD_SHARED_LIBS=ON
      -DENABLE_THREADS=${_fftw3_enable_threads}
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )
    set(_fftw3_extra_declare_args)
  endif()

  therock_cmake_subproject_declare(therock-fftw3
    USE_DIST_AMDGPU_TARGETS
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build_fftw3"
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    EXTERNAL_SOURCE_DIR "${_fftw3_external_source_dir}"
    ${_fftw3_extra_declare_args}
    CMAKE_ARGS
      ${_fftw3_common_cmake_args}
  )
  therock_cmake_subproject_provide_package(therock-fftw3 FFTW3 lib/cmake/fftw3)
  therock_cmake_subproject_activate(therock-fftw3)

  add_dependencies(therock-third-party therock-fftw3)

  therock_cmake_subproject_declare(therock-fftw3f
    USE_DIST_AMDGPU_TARGETS
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/build_fftw3f"
    BACKGROUND_BUILD
    EXCLUDE_FROM_ALL
    NO_MERGE_COMPILE_COMMANDS
    OUTPUT_ON_FAILURE
    EXTERNAL_SOURCE_DIR "${_fftw3_external_source_dir}"
    ${_fftw3_extra_declare_args}
    CMAKE_ARGS
      -DENABLE_FLOAT=ON
      ${_fftw3_common_cmake_args}
  )
  therock_cmake_subproject_provide_package(therock-fftw3f FFTW3f lib/cmake/fftw3f)
  therock_cmake_subproject_activate(therock-fftw3f)

  add_dependencies(therock-third-party therock-fftw3f)

  therock_provide_artifact(fftw3
    DESCRIPTOR artifact_fftw3.toml
    TARGET_NEUTRAL
    COMPONENTS
      dbg
      dev
      doc
      lib
      run
    SUBPROJECT_DEPS therock-fftw3 therock-fftw3f
  )
  return()
endif()

# Otherwise, this is the autotools sub-project build.
# SOURCE_DIR, ENABLE_THREADS, and ENABLE_FLOAT come from the super-project.
cmake_minimum_required(VERSION 3.25)
project(FFTW_BUILD VERSION 3.3.10 LANGUAGES NONE)

include(ProcessorCount)
ProcessorCount(_parallel_jobs)
if(NOT _parallel_jobs OR _parallel_jobs EQUAL 0)
  set(_parallel_jobs 1)
endif()

set(_configure_args
  --prefix=${CMAKE_INSTALL_PREFIX}
  --libdir=${CMAKE_INSTALL_PREFIX}/lib
  --enable-shared
  --disable-static
  --disable-fortran
)
if(ENABLE_THREADS)
  list(APPEND _configure_args --enable-threads)
endif()
if(ENABLE_FLOAT)
  list(APPEND _configure_args --enable-float)
  set(_package_name FFTW3f)
  set(_library_name fftw3f)
else()
  set(_package_name FFTW3)
  set(_library_name fftw3)
endif()

# cmake_DATA= skips the CMake package files from FFTW's Makefile.
# This project installs its own below.
add_custom_target(
  build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMAND "${SOURCE_DIR}/configure" ${_configure_args}
  COMMAND make -j ${_parallel_jobs}
  COMMAND make install cmake_DATA=
  VERBATIM
)

# Generate under cmake-package/. FFTW's Makefile writes a file of the same
# name in the build directory.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/FFTW3Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake-package/${_package_name}Config.cmake"
  @ONLY
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-package/${_package_name}Config.cmake"
  DESTINATION "lib/cmake/${_library_name}"
)
