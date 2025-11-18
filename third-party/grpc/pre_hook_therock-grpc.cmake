# Platform-specific build configuration for gRPC

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # C++ standard and PIC
  list(APPEND CMAKE_ARGS
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  )

  # Hide symbols to prevent ODR violations
  list(APPEND CMAKE_ARGS
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
  )

  # Exclude static library symbols from exports
  list(APPEND CMAKE_ARGS
    "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--exclude-libs,ALL"
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--exclude-libs,ALL"
    "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,--exclude-libs,ALL"
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # TODO: Add Windows-specific configuration
  message(WARNING "gRPC on Windows is not yet supported; Windows-specific flags not set")
endif()
