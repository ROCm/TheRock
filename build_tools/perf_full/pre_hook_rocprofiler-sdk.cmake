# Common repair, loaded after the generated toolchain and before SDK project().
# Derive actual HIP/device-lib prefixes from the same toolchain flags; never
# assume the imported-artifact directory matches the monolithic workspace.
separate_arguments(_perf_flags UNIX_COMMAND "${CMAKE_CXX_FLAGS_INIT} ${CMAKE_CXX_FLAGS}")
set(_perf_hip)
set(_perf_bitcode)
foreach(_flag IN LISTS _perf_flags)
  if(_flag MATCHES "^--hip-path=(.+)$")
    set(_perf_hip "${CMAKE_MATCH_1}")
  elseif(_flag MATCHES "^--hip-device-lib-path=(.+)$")
    set(_perf_bitcode "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(NOT EXISTS "${_perf_hip}/include/hip/hip_runtime.h" OR NOT IS_DIRECTORY "${_perf_bitcode}")
  message(FATAL_ERROR "Cannot verify generated HIP/device-library prefixes for SDK fixture repair")
endif()
set(CODEOBJ_HIP_PATH "${_perf_hip}" CACHE PATH "Actual generated HIP prefix" FORCE)
set(CODEOBJ_HIP_DEVICE_LIB_PATH "${_perf_bitcode}" CACHE PATH "Actual generated device-library prefix" FORCE)
file(WRITE "${CMAKE_BINARY_DIR}/perf-sdk-prefixes.txt"
  "hip=${_perf_hip}\nbitcode=${_perf_bitcode}\nflags=${CMAKE_CXX_FLAGS_INIT} ${CMAKE_CXX_FLAGS}\n")
