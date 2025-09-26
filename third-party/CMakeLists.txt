add_custom_target(therock-third-party)

# No-dep third party libraries (alphabetical)
add_subdirectory(boost)
add_subdirectory(eigen)
add_subdirectory(fmt)
add_subdirectory(googletest)
add_subdirectory(libdivide)
add_subdirectory(msgpack-cxx)
add_subdirectory(nlohmann-json)
add_subdirectory(yaml-cpp)
add_subdirectory(Catch2)
add_subdirectory(FunctionalPlus)
add_subdirectory(fftw)
add_subdirectory(fftwf)

# frugally-deep depends on eigen, FunctionalPlus and nlohmann-json
add_subdirectory(frugally-deep)

# spdlog depends on fmt.
add_subdirectory(spdlog)

# Host math libraries.
if(THEROCK_ENABLE_HOST_BLAS)
  add_subdirectory(host-blas)
endif()
if(THEROCK_ENABLE_HOST_SUITE_SPARSE)
  add_subdirectory(SuiteSparse)
endif()

# TODO: Relocate non header-only libraries here (i.e. boost, host-blas).
if(THEROCK_BUNDLE_SYSDEPS)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(sysdeps/linux)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_subdirectory(sysdeps/windows)
  endif()
endif()
