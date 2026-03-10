# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The opencl-cts CMakeCommon.txt installs to ${CMAKE_INSTALL_BINDIR}/$<CONFIG>,
# appending a config-name subdirectory (e.g. "Release") that we don't want.
# Flatten it at install time by moving the contents up one level.
install(CODE [[
  set(_cts_dir "${CMAKE_INSTALL_PREFIX}/share/opencl/opencl-cts")
  file(GLOB _config_subdirs LIST_DIRECTORIES true "${_cts_dir}/*")
  foreach(_subdir ${_config_subdirs})
    if(IS_DIRECTORY "${_subdir}")
      file(COPY "${_subdir}/" DESTINATION "${_cts_dir}")
      file(REMOVE_RECURSE "${_subdir}")
    endif()
  endforeach()
]])
