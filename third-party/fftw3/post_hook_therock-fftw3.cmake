# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Upstream exports relocatable targets, but its legacy directory variables embed
# the original install prefix. rocFFT uses those variables when building tests.
# Replace the generated config before upstream's install(FILES) copies it.
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/FFTW3Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/FFTW3${PREC_SUFFIX}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fftw3${PREC_SUFFIX}"
  PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
