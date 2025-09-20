add_custom_target(therock-third-party)

# No-dep third party libraries (alphabetical)
therock_add_subdirectory(boost)
therock_add_subdirectory(eigen)
therock_add_subdirectory(fmt)
therock_add_subdirectory(googletest)
therock_add_subdirectory(libdivide)
therock_add_subdirectory(msgpack-cxx)
therock_add_subdirectory(nlohmann-json)
therock_add_subdirectory(yaml-cpp)
therock_add_subdirectory(Catch2)
therock_add_subdirectory(FunctionalPlus)

# frugally-deep depends on eigen, FunctionalPlus and nlohmann-json
therock_add_subdirectory(frugally-deep)

# spdlog depends on fmt.
therock_add_subdirectory(spdlog)

# Host math libraries.
if(THEROCK_ENABLE_HOST_BLAS)
  therock_add_subdirectory(host-blas)
endif()
if(THEROCK_ENABLE_HOST_SUITE_SPARSE)
  therock_add_subdirectory(SuiteSparse)
endif()

# TODO: Relocate non header-only libraries here (i.e. boost, host-blas).
if(THEROCK_BUNDLE_SYSDEPS)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    therock_add_subdirectory(sysdeps/linux)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    therock_add_subdirectory(sysdeps/windows)
  endif()
endif()
