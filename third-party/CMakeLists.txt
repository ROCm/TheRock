# Subdirectory load order here is carefully balanced - see usage from the root
# CMakeLists.txt. Notably this file does *not* include the sysdeps subdirectory,
# which is expected to be included *first*. This allows for bootstrapping
# compiler/, so other projects in this directory can use the amd-llvm compiler
# toolchain if they so wish (most should eventually).

add_custom_target(therock-third-party)

# googletest is currently included explicitly ahead of the rest of this folder
# by the base CMakeLists.txt since since amdsmi in base/ depends on it and
# base/ is included before compiler/. To resolve that, we may want to move
# amdsmi from base/ to core/, after which we can turn googletest back into a
# typical third-party package.
# add_subdirectory(googletest)
add_dependencies(therock-third-party therock-googletest)

# No-dep third party libraries (alphabetical)
add_subdirectory(boost)
add_subdirectory(eigen)
add_subdirectory(flatbuffers)
add_subdirectory(fmt)
add_subdirectory(libdivide)
add_subdirectory(msgpack-cxx)
add_subdirectory(nlohmann-json)
add_subdirectory(simde)
add_subdirectory(yaml-cpp)
add_subdirectory(Catch2)
add_subdirectory(FunctionalPlus)
add_subdirectory(fftw3)

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
