# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The upstream wrapper writes libraries into its source directory. Keep all
# outputs in the build tree so different configurations can share sources.
set_target_properties(fftw3 PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt"
  DESTINATION share/licenses/fftz
)
