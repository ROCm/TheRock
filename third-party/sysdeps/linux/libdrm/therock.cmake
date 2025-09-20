# When included in TheRock, we download sources and set up the sub-project.
set(_source_dir "${THEROCK_CURRENT_BINARY_DIR}/source")
set(_download_stamp "${_source_dir}/download.stamp")

therock_subproject_fetch(therock-libdrm-sources
  SOURCE_DIR "${_source_dir}"
  # Originally mirrored from: "https://gitlab.freedesktop.org/mesa/drm/-/archive/libdrm-2.4.124/drm-libdrm-2.4.124.tar.bz2"
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/drm-libdrm-2.4.124.tar.bz2"
  URL_HASH "SHA256=18e66044e0542040614a7904b6a2c0e5249a81e705fe9ba5a1cc2e5df11416e6"
  TOUCH "${_download_stamp}"
)

therock_cmake_subproject_declare(therock-libdrm
  EXTERNAL_SOURCE_DIR .
  BINARY_DIR build
  NO_MERGE_COMPILE_COMMANDS
  BACKGROUND_BUILD
  OUTPUT_ON_FAILURE
  CMAKE_ARGS
    "-DSOURCE_DIR=${_source_dir}"
    "-DPATCHELF=${PATCHELF}"
    "-DMESON_BUILD=${MESON_BUILD}"
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
therock_cmake_subproject_activate(therock-libdrm)

therock_test_validate_shared_lib(
  PATH build/dist/lib/rocm_sysdeps/lib
  LIB_NAMES libdrm.so libdrm_amdgpu.so
)
