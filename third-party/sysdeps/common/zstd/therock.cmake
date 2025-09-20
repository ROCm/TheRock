set(_source_dir "${THEROCK_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(therock-zstd-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/zstd-1.5.7.tar.gz"
  URL_HASH "SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-zstd
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
therock_cmake_subproject_provide_package(therock-zstd zstd lib/rocm_sysdeps/lib/cmake/zstd)
therock_cmake_subproject_activate(therock-zstd)

therock_test_validate_shared_lib(
    PATH build/dist/lib/rocm_sysdeps/lib
    LIB_NAMES libzstd.so
)
