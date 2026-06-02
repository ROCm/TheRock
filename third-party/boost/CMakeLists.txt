# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Installs a limited subset of boost sufficient to build ROCM subprojects.
# Currently, this includes:
#   atomic
#   filesystem
#   multi_index
#   system
#   Static libraries only
#   Release build
#   Multithreaded
# See cmake_project/CMakeLists.txt for details.
set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(boost-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.bz2"
  # with `libs\wave\test\testwave\testfiles\utf8-test*` removed.
  URL "https://archives.boost.io/release/1.91.0/source/boost_1_91_0.tar.bz2"
  URL_HASH "SHA256=de5e6b0e4913395c6bdfa90537febd9028ea4c0735d2cdb0cd9b45d5f51264f5"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-boost
  EXTERNAL_SOURCE_DIR "cmake_project"
  USE_DIST_AMDGPU_TARGETS
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  BACKGROUND_BUILD
  OUTPUT_ON_FAILURE
  CMAKE_ARGS
    "-DBOOST_SOURCE_DIR=${_source_dir}"
    "-DTHEROCK_BOOST_LIBRARIES=atomic,filesystem"
  EXTRA_DEPENDS
    "${_download_stamp}"
)
# Important: Version number here must match the version downloaded above.
# Check the install directory if in doubt (i.e. run `ninja therock-boost` and
# see what the install layout is).
set(_boost_version 1.91.0)
# Advertise the Boost meta package and each supported sub-library. Must be
# kept in sync with above library list.
therock_cmake_subproject_provide_package(therock-boost Boost "lib/cmake/Boost-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_atomic "lib/cmake/boost_atomic-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_filesystem "lib/cmake/boost_filesystem-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_headers "lib/cmake/boost_headers-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_system "lib/cmake/boost_system-${_boost_version}")
therock_cmake_subproject_activate(therock-boost)

add_dependencies(therock-third-party therock-boost)
