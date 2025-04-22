# Disable INT128 support on Windows to avoid cross linkage issues when building
# with clang but depending on MSVC built support libraries:
# https://github.com/ROCm/TheRock/issues/405
if(WIN32)
  message(STATUS "Disabling INT128 support on Windows")
  add_compile_definitions("ROCPRIM_HAS_INT128_SUPPORT=0")
endif()
