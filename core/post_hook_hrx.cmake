# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

foreach(_target ${THEROCK_EXECUTABLE_TARGETS})
  if(_target MATCHES "^hrx_cts_")
    set_target_properties("${_target}" PROPERTIES
      THEROCK_INSTALL_RPATH_ORIGIN "lib/hrx/share/hrx-cts")
  endif()
endforeach()
