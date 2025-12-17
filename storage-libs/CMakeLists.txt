# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# hipFile CMake support for TheRock

if(THEROCK_ENABLE_HIPFILE)
  if(WIN32)
    message(FATAL_ERROR "hipFile is not supported on Win32")
  endif() 
endif()
