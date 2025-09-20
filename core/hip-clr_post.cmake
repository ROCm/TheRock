# See: https://github.com/ROCm/TheRock/issues/21
if(HIPCC_BIN_DIR)
  message(FATAL_ERROR "The legacy HIPCC_BIN_DIR was somehow set, indicating a bug in the clr CMake files: ${HIPCC_BIN_DIR}")
endif()
