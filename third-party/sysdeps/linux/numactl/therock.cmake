# When included in TheRock, we download sources and set up the sub-project.
set(_source_dir "${THEROCK_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(therock-numactl-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://github.com/numactl/numactl/releases/download/v2.0.19/numactl-2.0.19.tar.gz"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/numactl-2.0.19.tar.gz"
  URL_HASH "SHA256=f2672a0381cb59196e9c246bf8bcc43d5568bc457700a697f1a1df762b9af884"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-numactl
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
therock_cmake_subproject_provide_package(therock-numactl NUMA lib/rocm_sysdeps/lib/cmake/NUMA)
therock_cmake_subproject_activate(therock-numactl)

therock_test_validate_shared_lib(
  PATH build/dist/lib/rocm_sysdeps/lib
  LIB_NAMES libnuma.so
)
