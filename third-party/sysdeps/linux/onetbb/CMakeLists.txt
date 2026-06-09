# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # When included in TheRock, we download sources and set up the sub-project.
    set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
    set(_download_stamp "${_source_dir}/download.stamp")

    therock_subproject_fetch(therock-onetbb-sources
      SOURCE_DIR "${_source_dir}"
      # Originally downloaded from: https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2023.0.0.tar.gz
      URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/oneTBB-2023.0.0.tar.gz"
      URL_HASH "SHA256=f8767b971ec6aea25dde58ae0f593e94e7aa75a739a86f67967012f69e2199b1"
      PATCH_COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build_tools/patch_third_party_source.py ${CMAKE_SOURCE_DIR}/patches/third-party/onetbb

      TOUCH "${_download_stamp}"
    )

    therock_cmake_subproject_declare(therock-onetbb
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
    therock_cmake_subproject_provide_package(therock-onetbb TBB lib/rocm_sysdeps/lib/cmake/TBB)
    therock_cmake_subproject_activate(therock-onetbb)

    # Test key headers
    set(ONETBB_HEADERS "tbb/tbb.h" "oneapi/tbb.h")
    foreach(onetbb_header ${ONETBB_HEADERS})
      string(REPLACE "/" "_" header_test_name "${onetbb_header}")
      add_test(
        NAME onetbb_header_test:${header_test_name}
        COMMAND ls "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/include/${onetbb_header}"
      )
    endforeach()

    return()
endif()

# Otherwise, this is the sub-project build.
cmake_minimum_required(VERSION 3.25)
project(ONETBB_BUILD VERSION "2023.0.0")

if(NOT PATCHELF)
  message(FATAL_ERROR "Missing PATCHELF from super-project")
endif()

add_custom_target(
  onetbb_build ALL
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"

  COMMAND
    "${CMAKE_COMMAND}" -E rm -rf -- "${CMAKE_INSTALL_PREFIX}"
  COMMAND
    "${CMAKE_COMMAND}"
      "-G${CMAKE_GENERATOR}"
      "-S${SOURCE_DIR}"
      "-B${CMAKE_CURRENT_BINARY_DIR}/b"
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
      -DCMAKE_INSTALL_LIBDIR=lib
      -DBUILD_SHARED_LIBS=OFF
      -DTBB_TEST=OFF
      -DTBB_EXAMPLES=OFF
      -DTBB_STRICT=OFF
      -DTBB_INSTALL=ON
      -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON
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

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/tbb-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TBBConfig.cmake
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TBBConfig.cmake" DESTINATION lib/cmake/TBB)
