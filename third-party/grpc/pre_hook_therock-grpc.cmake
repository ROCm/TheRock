# Platform-specific build configuration for gRPC

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # C++ standard and PIC
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # Hide symbols to prevent ODR violations
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

  # Exclude static library symbols from exports
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Wl,--exclude-libs,ALL")
  string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,--exclude-libs,ALL")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS " -Wl,--exclude-libs,ALL")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # TODO: Add Windows-specific configuration when gRPC is needed on Windows
  message(STATUS "gRPC on Windows: configuration pending")
endif()
