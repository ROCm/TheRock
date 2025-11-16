if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # gRPC static library for RDC
  if(NOT DEFINED THEROCK_GRPC_ZLIB_STAGE_LIB_DIR OR "${THEROCK_GRPC_ZLIB_STAGE_LIB_DIR}" STREQUAL "")
    message(FATAL_ERROR "THEROCK_GRPC_ZLIB_STAGE_LIB_DIR must be provided by the super-project before configuring gRPC")
  endif()
  set(_therock_zlib_stage_lib_dir "${THEROCK_GRPC_ZLIB_STAGE_LIB_DIR}")

  set(_source_dir "${CMAKE_CURRENT_BINARY_DIR}/source")
  set(_download_stamp "${_source_dir}/download.stamp")

  therock_subproject_fetch(therock-grpc-sources
    CMAKE_PROJECT
    SOURCE_DIR "${_source_dir}"
    GIT_REPOSITORY "https://github.com/grpc/grpc.git"
    GIT_TAG "v1.67.1"
    GIT_SHALLOW ON
    GIT_SUBMODULES_RECURSE ON
    TOUCH "${_download_stamp}"
  )

  set(_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/build")

  therock_cmake_subproject_declare(therock-grpc
    EXTERNAL_SOURCE_DIR "${_source_dir}"
    BINARY_DIR "${_binary_dir}"
    NO_MERGE_COMPILE_COMMANDS
    BACKGROUND_BUILD
    OUTPUT_ON_FAILURE

    CMAKE_ARGS
      -DCMAKE_CXX_STANDARD=17
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DBUILD_SHARED_LIBS=OFF
      -DgRPC_BUILD_SHARED_LIBS=OFF
      -DgRPC_PREFER_STATIC_LIBS=ON
      -Dprotobuf_BUILD_SHARED_LIBS=OFF
      -DCMAKE_CXX_VISIBILITY_PRESET=hidden
      -DCMAKE_C_VISIBILITY_PRESET=hidden
      -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
      "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,--exclude-libs,ALL"
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--exclude-libs,ALL"
      "-DCMAKE_MODULE_LINKER_FLAGS=-Wl,--exclude-libs,ALL"
      "-DgRPC_ZLIB_PROVIDER=package"
      "-DgRPC_ABSL_PROVIDER=module"
      "-DgRPC_CARES_PROVIDER=module"
      "-DgRPC_RE2_PROVIDER=module"
      "-DgRPC_SSL_PROVIDER=module"
      "-DgRPC_PROTOBUF_PROVIDER=module"
      "-DgRPC_BUILD_TESTS=OFF"
      "-DgRPC_BUILD_CSHARP_EXT=OFF"
      "-DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF"
      "-DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF"
      "-DgRPC_INSTALL=ON"
      "-DCMAKE_INSTALL_LIBDIR=lib"
      "-DCMAKE_INSTALL_RPATH=\\$ORIGIN:\\$ORIGIN/../lib"
      "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
      "-DCMAKE_SKIP_BUILD_RPATH=OFF"

    RUNTIME_DEPS
      therock-zlib

    INSTALL_DESTINATION
      lib/rocm_sysdeps

    INTERFACE_INCLUDE_DIRS
      lib/rocm_sysdeps/include

    INTERFACE_LINK_DIRS
      lib/rocm_sysdeps/lib

    INTERFACE_INSTALL_RPATH_DIRS
      lib/rocm_sysdeps/lib

    INTERFACE_PKG_CONFIG_DIRS
      lib/rocm_sysdeps/lib/pkgconfig

    EXTRA_DEPENDS
      "${_download_stamp}"

    CMAKE_INCLUDES
      "${CMAKE_CURRENT_SOURCE_DIR}/post_hook_therock-grpc.cmake"
  )
  set(THEROCK_GRPC_ZLIB_STAGE_LIB_DIR "${_therock_zlib_stage_lib_dir}")
  therock_cmake_subproject_provide_package(therock-grpc gRPC lib/rocm_sysdeps/lib/cmake/grpc)
  therock_cmake_subproject_provide_package(therock-grpc protobuf lib/rocm_sysdeps/lib/cmake/protobuf)
  therock_cmake_subproject_activate(therock-grpc)

  # Static library validation tests
  foreach(lib_name libgrpc++.a libgrpc.a libprotobuf.a)
    add_test(
      NAME therock-validate-static-lib-${lib_name}
      COMMAND
        "${CMAKE_CURRENT_SOURCE_DIR}/validate_static_library.sh"
          "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/lib/${lib_name}"
    )
  endforeach()

  # Comprehensive linking test (all libraries together)
  add_test(
    NAME therock-validate-grpc-static-libs-linkage
    COMMAND
      "${CMAKE_CURRENT_SOURCE_DIR}/validate_static_library.sh"
        "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/lib/libgrpc++.a"
        "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/lib/libgrpc.a"
        "${CMAKE_CURRENT_BINARY_DIR}/build/dist/lib/rocm_sysdeps/lib/libprotobuf.a"
  )
  set_tests_properties(therock-validate-grpc-static-libs-linkage PROPERTIES
    LABELS "grpc;static-library;linkage"
  )

  # Note: Static libraries must export public API symbols (T) for linking (~451 global symbols).

  return()
endif()

cmake_minimum_required(VERSION 3.25)
project(therock-grpc-wrapper NONE)
message(FATAL_ERROR
  "third-party/grpc/CMakeLists.txt is intended to be included via TheRock super-project.\n"
  "Please configure and build gRPC through TheRock instead.")
