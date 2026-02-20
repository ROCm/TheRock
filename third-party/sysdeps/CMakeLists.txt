# TODO: Relocate non header-only libraries here (i.e. boost, host-blas).
if(THEROCK_BUNDLE_SYSDEPS)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(linux)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_subdirectory(windows)
  endif()
endif()
