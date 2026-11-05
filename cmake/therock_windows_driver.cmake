# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Apply the driver variant before finalizing the generated feature dependencies.
if(WIN32 AND THEROCK_FLAG_WINDOWS_DRIVER_BUILD)
  if(THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS)
    message(FATAL_ERROR
      "Windows driver builds require -DTHEROCK_FLAG_KPACK_SPLIT_ARTIFACTS=OFF "
      "because the driver HIP runtime does not support kpack."
    )
  endif()
  set(THEROCK_ENABLE_KPACK OFF)
  list(REMOVE_ITEM THEROCK_REQUIRES_HIP_RUNTIME KPACK)
endif()
