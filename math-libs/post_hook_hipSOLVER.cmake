list(APPEND CMAKE_MODULE_PATH "${THEROCK_SOURCE_DIR}/cmake")
include(therock_rpath)

therock_set_install_rpath(
  TARGETS
    hipsolver
  PATHS
    .
    host-math/lib
)
