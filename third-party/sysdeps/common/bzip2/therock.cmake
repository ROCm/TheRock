set(_source_dir "${THEROCK_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(therock-bzip2-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://sourceware.org/pub/bzip2/bzip2-1.0.8.tar.gz"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/bzip2-1.0.8.tar.gz"
  URL_HASH "SHA512=083f5e675d73f3233c7930ebe20425a533feedeaaa9d8cc86831312a6581cefbe6ed0d08d2fa89be81082f2a5abdabca8b3c080bf97218a1bd59dc118a30b9f3"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-bzip2
  EXTERNAL_SOURCE_DIR .
  BINARY_DIR build
  NO_MERGE_COMPILE_COMMANDS
  BACKGROUND_BUILD
  OUTPUT_ON_FAILURE
  CMAKE_ARGS
    "-DSOURCE_DIR=${_source_dir}"
  INSTALL_DESTINATION
    lib/rocm_sysdeps
  INTERFACE_PROGRAM_DIRS
    lib/rocm_sysdeps/bin
  INTERFACE_LINK_DIRS
    lib/rocm_sysdeps/lib
  INTERFACE_INSTALL_RPATH_DIRS
    lib/rocm_sysdeps/lib
  INTERFACE_PKG_CONFIG_DIRS
    lib/rocm_sysdeps/lib/pkgconfig
  EXTRA_DEPENDS
    "${_download_stamp}"
)
therock_cmake_subproject_provide_package(therock-bzip2 BZip2 lib/rocm_sysdeps/lib/cmake/BZip2)
therock_cmake_subproject_activate(therock-bzip2)

therock_test_validate_shared_lib(
  PATH build/dist/lib/rocm_sysdeps/lib
  LIB_NAMES libbz2.so
)
return()
