therock_subproject_fetch(therock-dyninst-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/dyninst/dyninst/archive/refs/tags/v13.0.0.tar.gz
  # TODO: Replace with S3 bucket URL
  URL https://github.com/dyninst/dyninst/archive/refs/tags/v13.0.0.tar.gz
  URL_HASH SHA256=1bc48d26478b677a6c090c25586a447507bd1b4cf88d369bd61820005ce1be39
)

# Still a work in progress - need to figure out why the boost libraries are not being found.
therock_cmake_subproject_declare(therock-dyninst
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BUILD_DEPS
    therock-boost
    therock-tbb
    therock-elfutils
  CMAKE_ARGS
    -DBoost_NO_SYSTEM_PATHS=NO
    -DBoost_ROOT_DIR="${CMAKE_CURRENT_BINARY_DIR}/../boost/cmake_project/stage"
    -DBoost_LIBRARY_DIRS="${CMAKE_CURRENT_BINARY_DIR}/../boost/cmake_project/stage/lib"
    -DBoost_INCLUDE_DIRS="${CMAKE_CURRENT_BINARY_DIR}/../boost/cmake_project/stage/include"
    -DTBB_ROOT_DIR="${CMAKE_CURRENT_BINARY_DIR}/../tbb/stage"
    -DElfUtils_ROOT_DIR="${CMAKE_CURRENT_BINARY_DIR}/../sysdeps/linux/elfutils/build/stage/lib/rocm_sysdeps"
    -DLibIberty_ROOT_DIR="/opt/rh/gcc-toolset-13/root/usr"
)
therock_cmake_subproject_provide_package(therock-dyninst dyninst lib/cmake)
therock_cmake_subproject_activate(therock-dyninst)

add_dependencies(therock-third-party therock-dyninst)
