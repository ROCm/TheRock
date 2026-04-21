# TheRock OpenMP compatibility shim - MODULE-mode entry point.
#
# When TheRock's dependency provider resolves an explicit find_package(OpenMP
# MODULE) call, it sets CMAKE_MODULE_PATH to this directory. Defer to the
# CONFIG-mode shim that sits next to this file, which includes LLVM's
# openmp-config.cmake and adds Module-compat targets.

include("${CMAKE_CURRENT_LIST_DIR}/openmp-config.cmake")
