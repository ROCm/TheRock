# Subdirectory load order here is carefully balanced - see usage from the root
# CMakeLists.txt. Notably this file does *not* include the sysdeps subdirectory,
# which is expected to be included *first*. This allows for bootstrapping
# compiler/, so other projects in this directory can use the amd-llvm compiler
# toolchain if they so wish (most should eventually).

add_custom_target(therock-third-party)

# Third-party libraries built with the system toolchain.
add_subdirectory(eigen)
if(THEROCK_ENABLE_HOST_BLAS)
  add_subdirectory(host-blas)
endif()

# Third-party libraries built with the amd-llvm compiler toolchain.
if(THEROCK_ENABLE_COMPILER)
  # No-dep libraries (alphabetical)
  add_subdirectory(boost)
  add_subdirectory(Catch2)
  add_subdirectory(fftw3)
  add_subdirectory(flatbuffers)
  add_subdirectory(fmt)
  add_subdirectory(FunctionalPlus)
  add_subdirectory(googletest)
  add_subdirectory(libdivide)
  add_subdirectory(msgpack-cxx)
  add_subdirectory(nlohmann-json)
  add_subdirectory(simde)
  add_subdirectory(yaml-cpp)

  # frugally-deep depends on eigen, FunctionalPlus and nlohmann-json
  add_subdirectory(frugally-deep)

  # spdlog depends on fmt.
  add_subdirectory(spdlog)

  # Host math library (SuiteSparse uses amd-llvm toolchain).
  if(THEROCK_ENABLE_HOST_SUITE_SPARSE)
    add_subdirectory(SuiteSparse)
  endif()

  # gRPC: Static library for RDC (built on-demand when depended upon)
  # See: docs/rfcs/RFC0007-rdc-therock-integration.md
  add_subdirectory(grpc)
endif()
