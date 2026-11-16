# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# libhipcxx_tests builds libhipcxx's own source tree (in-tree headers), but its
# configure finds the staged rocthrust/hipcub, whose exported targets link
# libhipcxx::libhipcxx (hipCCL was built against the installed libhipcxx). That
# target is only defined by the installed package, not by the source tree, so
# CMake fails generating roc::rocthrust. Provide an empty placeholder: the tests
# get their headers from the in-tree libcudacxx::libcudacxx target instead.
if(NOT TARGET libhipcxx::libhipcxx)
  add_library(libhipcxx::libhipcxx INTERFACE IMPORTED GLOBAL)
endif()
