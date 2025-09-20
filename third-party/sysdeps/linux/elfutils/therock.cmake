set(_source_dir "${THEROCK_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(therock-elfutils-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://sourceware.org/elfutils/ftp/0.192/elfutils-0.192.tar.bz2"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/elfutils-0.192.tar.bz2"
  URL_HASH "SHA512=543188f5f2cfe5bc7955a878416c5f252edff9926754e5de0c6c57b132f21d9285c9b29e41281e93baad11d4ae7efbbf93580c114579c182103565fe99bd3909"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-elfutils
  EXTERNAL_SOURCE_DIR .
  BINARY_DIR build
  NO_MERGE_COMPILE_COMMANDS
  BACKGROUND_BUILD
  OUTPUT_ON_FAILURE
  CMAKE_ARGS
    "-DSOURCE_DIR=${_source_dir}"
    "-DPATCHELF=${PATCHELF}"
    "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}"
  RUNTIME_DEPS
    therock-bzip2
    therock-zlib
    therock-zstd
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
therock_cmake_subproject_provide_package(therock-elfutils LibElf lib/rocm_sysdeps/lib/cmake/LibElf)
therock_cmake_subproject_provide_package(therock-elfutils libdw lib/rocm_sysdeps/lib/cmake/libdw)
therock_cmake_subproject_activate(therock-elfutils)

therock_test_validate_shared_lib(
  PATH build/dist/lib/rocm_sysdeps/lib
  LIB_NAMES libelf.so libdw.so libasm.so
)
