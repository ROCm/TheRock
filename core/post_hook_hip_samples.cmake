# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install HIP samples source files to share/hip/samples/
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION share/hip/samples
)
