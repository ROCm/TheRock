# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RPP configures both HOST and HIP test programs unconditionally. The host-TSAN
# lane installs the shared test sources and builds only the exact HOST targets
# in its no-device runner, so keep every device-side test executable out of the
# component's default build graph. Normal release and sanitizer variants retain
# the upstream behavior.
if(THEROCK_SANITIZER STREQUAL "HOST_TSAN")
  foreach(_target
      Tensor_image_hip
      Tensor_misc_hip
      Tensor_voxel_hip
      Tensor_audio_hip)
    if(TARGET "${_target}")
      set_property(TARGET "${_target}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    endif()
  endforeach()
  message(STATUS "RPP host-TSAN: excluded device-side HIP test executables")
endif()
