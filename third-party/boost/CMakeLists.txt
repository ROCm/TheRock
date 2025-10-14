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
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/boost_1_87_0-1.tar.bz2"
  URL_HASH "SHA256=0ba4085a0beb3b9c57fa617d09110c79e977a9944750cd93fb17d82807a8c64c"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-boost
  EXTERNAL_SOURCE_DIR "cmake_project"
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  BACKGROUND_BUILD
  OUTPUT_ON_FAILURE
  CMAKE_ARGS
    "-DBOOST_SOURCE_DIR=${_source_dir}"
    "-DTHEROCK_BOOST_LIBRARIES=atomic,filesystem,multi_index,system"
  EXTRA_DEPENDS
    "${_download_stamp}"
)
# Important: Version number here must match the version downloaded above.
# Check the install directory if in doubt (i.e. run `ninja therock-boost` and
# see what the install layout is).
set(_boost_version 1.87.0)
# Advertise the Boost meta package and each supported sub-library. Must be
# kept in sync with above library list.
therock_cmake_subproject_provide_package(therock-boost Boost "lib/cmake/Boost-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_atomic "lib/cmake/boost_atomic-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_filesystem "lib/cmake/boost_filesystem-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_headers "lib/cmake/boost_headers-${_boost_version}")
therock_cmake_subproject_provide_package(therock-boost boost_system "lib/cmake/boost_system-${_boost_version}")
therock_cmake_subproject_activate(therock-boost)

add_dependencies(therock-third-party therock-boost)
