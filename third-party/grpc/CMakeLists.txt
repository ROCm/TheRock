# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# gRPC static library for RDC
set(THEROCK_GRPC_VERSION "1.78.1" CACHE INTERNAL "gRPC version built by TheRock")

therock_subproject_fetch(therock-grpc-sources
  CMAKE_PROJECT
  # Originally mirrored from: https://github.com/grpc/grpc.git
  URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/grpc-v1.78.1.tar.gz"
  URL_HASH "SHA256=96e1d4c82735a6867b9045d782d3f4f7477224496fec5386ca3adae1f2059796"
)

set(_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/build")

therock_cmake_subproject_declare(therock-grpc
  USE_DIST_AMDGPU_TARGETS
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  BINARY_DIR "${_binary_dir}"
  NO_MERGE_COMPILE_COMMANDS
  BACKGROUND_BUILD
  OUTPUT_ON_FAILURE

  CMAKE_ARGS
    # Static linking configuration
    -DBUILD_SHARED_LIBS=OFF
    -DgRPC_BUILD_SHARED_LIBS=OFF
    -DgRPC_PREFER_STATIC_LIBS=ON
    -Dprotobuf_BUILD_SHARED_LIBS=OFF

    # Dependency providers
    -DgRPC_ZLIB_PROVIDER=package
    -DgRPC_ABSL_PROVIDER=module
    -DgRPC_CARES_PROVIDER=module
    -DgRPC_RE2_PROVIDER=module
    -DgRPC_SSL_PROVIDER=module
    -DgRPC_PROTOBUF_PROVIDER=module

    # Disable tests and unnecessary language plugins
    -DgRPC_BUILD_TESTS=OFF
    -DgRPC_BUILD_CSHARP_EXT=OFF
    -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF
    -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF
    -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF
    -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF
    -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF
    -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF

    # Installation configuration
    -DgRPC_INSTALL=ON
    -DCMAKE_INSTALL_LIBDIR=lib
    "-DCMAKE_INSTALL_RPATH=\\$ORIGIN:\\$ORIGIN/../lib"
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
    -DCMAKE_SKIP_BUILD_RPATH=OFF
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

  RUNTIME_DEPS
    ${THEROCK_BUNDLED_ZLIB}
)
therock_cmake_subproject_provide_package(therock-grpc gRPC lib/cmake/grpc)
therock_cmake_subproject_provide_package(therock-grpc protobuf lib/cmake/protobuf)
therock_cmake_subproject_activate(therock-grpc)
add_dependencies(therock-third-party therock-grpc)
