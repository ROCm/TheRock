set(_source_dir "${THEROCK_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(therock-sqlite3-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://www.sqlite.org/2025/sqlite-amalgamation-3490100.zip"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/sqlite-amalgamation-3490100.zip"
  URL_HASH "SHA256=6cebd1d8403fc58c30e93939b246f3e6e58d0765a5cd50546f16c00fd805d2c3"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-sqlite3
  EXTERNAL_SOURCE_DIR .
  BINARY_DIR build
  NO_MERGE_COMPILE_COMMANDS
  BACKGROUND_BUILD
  OUTPUT_ON_FAILURE
  CMAKE_ARGS
    "-DSOURCE_DIR=${_source_dir}"
  INSTALL_DESTINATION
    lib/rocm_sysdeps
  INTERFACE_LINK_DIRS
    lib/rocm_sysdeps/lib
  INTERFACE_PKG_CONFIG_DIRS
    lib/rocm_sysdeps/lib/pkgconfig
  EXTRA_DEPENDS
    "${_download_stamp}"
)
therock_cmake_subproject_provide_package(therock-sqlite3 SQLite3 lib/rocm_sysdeps/lib/cmake/SQLite3)
therock_cmake_subproject_activate(therock-sqlite3)

therock_test_validate_shared_lib(
  PATH build/dist/lib/rocm_sysdeps/lib
  LIB_NAMES libsqlite3.so
)
