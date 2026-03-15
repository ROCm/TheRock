# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install HIP samples source files to share/hip/samples/
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
  DESTINATION share/hip/samples
  PATTERN "build" EXCLUDE
)

# The upstream samples CMakeLists.txt marks executables as EXCLUDE_FROM_ALL.
# Override that so they build normally, and add install rules to place
# the built executables into share/hip/samples/bin/.
# THEROCK_EXECUTABLE_TARGETS is populated by therock_global_post_subproject.cmake
# before this post-hook runs.
foreach(_target IN LISTS THEROCK_EXECUTABLE_TARGETS)
  set_target_properties(${_target} PROPERTIES EXCLUDE_FROM_ALL FALSE)
  install(TARGETS ${_target} RUNTIME DESTINATION share/hip/samples/bin)
endforeach()
