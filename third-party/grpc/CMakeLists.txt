# gRPC static library for RDC
therock_subproject_fetch(therock-grpc-sources
  CMAKE_PROJECT
  # TODO: Mirror this repository to S3 for better availability and faster downloads
  GIT_REPOSITORY "https://github.com/grpc/grpc.git"
  GIT_TAG "v1.67.1"
  GIT_SHALLOW ON
  GIT_SUBMODULES_RECURSE ON
)

set(_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/build")

therock_cmake_subproject_declare(therock-grpc
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

  RUNTIME_DEPS
    therock-zlib
)
therock_cmake_subproject_provide_package(therock-grpc gRPC lib/cmake/grpc)
therock_cmake_subproject_provide_package(therock-grpc protobuf lib/cmake/protobuf)
therock_cmake_subproject_activate(therock-grpc)
add_dependencies(therock-third-party therock-grpc)
