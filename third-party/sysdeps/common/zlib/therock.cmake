set(_source_dir "${THEROCK_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(therock-zlib-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://www.zlib.net/zlib-1.3.1.tar.gz"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/zlib-1.3.1.tar.gz"
  URL_HASH "SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-zlib
  EXTERNAL_SOURCE_DIR .
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
therock_cmake_subproject_provide_package(therock-zlib ZLIB lib/rocm_sysdeps/lib/cmake/ZLIB)
therock_cmake_subproject_activate(therock-zlib)

therock_test_validate_shared_lib(
  PATH build/dist/lib/rocm_sysdeps/lib
  LIB_NAMES libz.so
)
