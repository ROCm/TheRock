add_custom_target(therock-third-party)

# No-dep third party libraries (alphabetical)
add_subdirectory(boost)
add_subdirectory(eigen)
add_subdirectory(flatbuffers)
add_subdirectory(fmt)
add_subdirectory(googletest)
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

# TODO: Relocate non header-only libraries here (i.e. boost, host-blas).
if(THEROCK_BUNDLE_SYSDEPS)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(sysdeps/linux)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_subdirectory(sysdeps/windows)
  endif()
endif()

# gRPC: RDC-exclusive dependency (static library)
# - Built for RDC standalone mode only (Linux)
# - Depends on bundled zlib from sysdeps
# - See: docs/rfcs/RFC0007-rdc-therock-integration.md
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND THEROCK_BUNDLE_SYSDEPS)
  get_target_property(_therock_zlib_stage ${THEROCK_BUNDLED_ZLIB} THEROCK_STAGE_DIR)
  if(NOT _therock_zlib_stage)
    message(FATAL_ERROR "Failed to locate stage directory for ${THEROCK_BUNDLED_ZLIB}")
  endif()
  set(THEROCK_GRPC_ZLIB_STAGE_LIB_DIR "${_therock_zlib_stage}/lib/rocm_sysdeps/lib" CACHE INTERNAL
    "Staged zlib lib directory for gRPC post-install patch" FORCE)
  add_subdirectory(grpc)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Skipping bundled gRPC build (requires THEROCK_BUNDLE_SYSDEPS=ON for zlib dependency)")
endif()
