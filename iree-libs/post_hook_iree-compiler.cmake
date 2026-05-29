# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# IREE's compiler tools link libIREECompiler.dylib via @rpath, but the upstream
# install rules do not consistently preserve an install RPATH on Darwin. Feed
# this through TheRock's global post hook so it is not overwritten by the
# default install-RPATH pass.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  list(APPEND THEROCK_PRIVATE_INSTALL_RPATH_DIRS "lib")
endif()

set(_therock_iree_compiler_tools
  iree-compile
  iree-link
  iree-lld
  iree-mlir-lsp-server
  iree-opt
  iree-reduce
  iree-run-mlir
)

foreach(_therock_tool ${_therock_iree_compiler_tools})
  if(TARGET "${_therock_tool}")
    therock_set_install_rpath(TARGETS "${_therock_tool}" PATHS "../lib")
  endif()
endforeach()
